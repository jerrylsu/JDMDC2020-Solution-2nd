# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
import random
from argparse import ArgumentParser
from tqdm import tqdm
from pprint import pformat
import warnings
import json
import re
import torch
import torch.nn.functional as F
from gpt_model import GPT2LMHeadModel, BertTokenizer
from featurizer.get_dataloader import SPECIAL_TOKENS, build_input_from_segments, get_dataloader

PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__))
TEST_DATA = os.path.join(PROJECT_FOLDER, 'data/output_data/gpt2_test.json')
# MODEL_CHECKPOINT = os.path.join(PROJECT_FOLDER, 'runs/model')
# MODEL_CHECKPOINT = os.path.join(PROJECT_FOLDER, 'runs/Sep02_13-11-14_5cc6919aa215_gpt2')
# MODEL_CHECKPOINT = os.path.join(PROJECT_FOLDER, 'runs/Sep07_15-03-23_5cc6919aa215_gpt2')   # base model
# MODEL_CHECKPOINT = os.path.join(PROJECT_FOLDER, 'runs/Sep07_15-07-21_5cc6919aa215_gpt2')   # medium model
MODEL_CHECKPOINT = os.path.join(PROJECT_FOLDER, 'runs/Sep10_12-46-23_5cc6919aa215_gpt2')     # base model with images
ONLINE_DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'online_test_data')


def online_test_postprocess(args, predictions):
    with open(args.online_test_questions) as fp:
        ques_items = json.load(fp)
    output_list = []
    for question, answer in zip(ques_items, predictions):
        result_dict = dict()
        result_dict['Id'] = question['Id']
        result_dict['Answer'] = answer
        output_list.append(result_dict)
    with open(args.online_test_answers, 'w') as fp:
        result = json.dumps(output_list, ensure_ascii=False, indent=2)
        fp.write(result)


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(persona, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(persona, history, current_output, tokenizer, args.input_max_length, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def beam_search(model, input_ids, token_type_ids, image_names, image_ids, args):
    outputs = model.generate(input_ids,
                             token_type_ids=token_type_ids,
                             input_images=image_names,
                             image_ids=image_ids,
                             num_beams=args.num_beams,
                             do_sample=False,
                             temperature=0.7,
                             top_k=0,
                             top_p=0.9,
                             max_length=args.max_length + input_ids.size(-1),
                             bos_token_id=0,
                             pad_token_id=1,
                             eos_token_ids=2,
                             num_return_sequences=1)
    outputs = outputs.data.cpu().numpy().tolist()
    return outputs


def run():
    parser = ArgumentParser()
    parser.add_argument("--test_data_file", type=str, default=TEST_DATA, help="Path or url of the dataset.")
    parser.add_argument("--output_file", type=str, default="", help="Path of response generated.")
    parser.add_argument("--online_test_questions", type=str, default=os.path.join(ONLINE_DATA_FOLDER, "test_questions.json"))
    parser.add_argument("--online_test_answers", type=str, default=os.path.join(ONLINE_DATA_FOLDER, "test_answers.json"))
    parser.add_argument("--image_path", type=str, default=os.path.join(ONLINE_DATA_FOLDER, "images_test"), help="Path of the images.")
    parser.add_argument("--model_checkpoint", type=str, default=MODEL_CHECKPOINT, help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', default=True, help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--input_max_length", type=int, default=256, help="Max length of input sentence")
    parser.add_argument("--num_beams", type=int, default=3, help="Beam num")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        logging.error("Loaded model checkpoint error!")
        return

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = BertTokenizer, GPT2LMHeadModel
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    model.eval()

    test_dataloader = get_dataloader(args, tokenizer)
    predictions = []
    for batch in tqdm(test_dataloader, ncols=80):
        batch = tuple(torch.tensor(input_data).to(args.device) if idx not in [2, 3] else input_data for idx, input_data in enumerate(batch))
        input_ids, token_type_ids, image_names, image_ids = batch
        if args.no_sample:
            outputs = beam_search(model, input_ids, token_type_ids, image_names, image_ids, args)
        else:
            with torch.no_grad():
                out_ids = sample_sequence(persona, history, tokenizer, model, args)
        for output in outputs:
            out_text = tokenizer.convert_ids_to_tokens(output[input_ids.size(-1):])
            out_text = ''.join(out_text)
            out_text = out_text.replace('|||', ' ')
            out_text = out_text.replace('<img>', '')
            out_text = out_text.replace('[UNK]', '')
            predictions.append(out_text)

    online_test_postprocess(args, predictions)


if __name__ == "__main__":
    run()
