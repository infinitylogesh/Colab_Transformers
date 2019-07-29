from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertConfig,BertForQuestionAnswering
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utils_squad import read_squad_examples,convert_examples_to_features,get_final_text,_compute_softmax
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended,_get_best_indexes)
from utils_squad import SquadExample
from run_squad import to_list
import collections
import torch


def load_artifacts(model_path):
    """ Loads pretrained model , tokenizer , config."""
    model_class = BertForQuestionAnswering
    model = model_class.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model.to("cpu")
    model.eval()
    return model,tokenizer,config

def read_squad_repl(context,question):
    """ Convert context and question into a SquadExample."""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    paragraph_text =context
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    qas_id = "id"
    question_text = question
    start_position = None
    end_position = None
    orig_answer_text = None
    is_impossible = False
    start_position = -1
    end_position = -1
    orig_answer_text = ""
    example = SquadExample(
        qas_id=qas_id,
        question_text=question_text,
        doc_tokens=doc_tokens,
        orig_answer_text=orig_answer_text,
        start_position=start_position,
        end_position=end_position,
        is_impossible=is_impossible)
    examples.append(example)
    return examples


def feature_extract(context,question,tokenizer):
    """ Converts context and questions to input features """
    examples = read_squad_repl(context,question)
    features = convert_examples_to_features(examples=examples,tokenizer=tokenizer,
                                                    max_seq_length=384,
                                                    doc_stride=128,
                                                    max_query_length=64,
                                                    is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_cls_index, all_p_mask)
    batch = tuple(t.to("cpu") for t in dataset[0])
    inputs = {'input_ids':     batch[0].unsqueeze(0),
                        'attention_mask': batch[1].unsqueeze(0),  # XLM don't use segment_ids
                        'token_type_ids': batch[2].unsqueeze(0)}
    example_indices = batch[3].unsqueeze(0)
    return inputs,features,examples,example_indices


def decode_ouput(outputs,features,examples,example_indices):
    """ Helps in decoding the model ouput to nbest probabilites and start , end word indexes """
    eval_feature = features[example_indices[0].item()]
    unique_id = int(eval_feature.unique_id)
    result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][0]),
                                    end_logits   = to_list(outputs[1][0]))
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
    start_indexes = _get_best_indexes(to_list(outputs[0][0]),20)
    end_indexes = _get_best_indexes(to_list(outputs[1][0]),20)
    feature = eval_feature
    prelim_predictions = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index not in feature.token_to_orig_map:
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if not feature.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > 30:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=0,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))
    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= 30:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = examples[0].doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, True, True)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))
   
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)
    return nbest_json,nbest_json[0]["text"]


# context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
# question = "Which NFL team represented the AFC at Super Bowl 50?"
# model_path = "pretrained/"
# model = load_model(model_path)
# inputs,features,examples,example_indices = feature_extract(context,question)
# ouputs = model(**inputs)
# nbest,best = decode_ouput(ouputs,features,examples,example_indices)
# print best