import torch
import os
import json
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from utils.utils import get_dataset
from tqdm import tqdm
import numpy as np
from featurizer.dialogue_dataset import DialoDataset, DialoImageDataset

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]", "<img>", "[PAD]"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids", "input_images_name", "input_images_id"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


def get_test_data(dataset_path, tokenizer):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    if isinstance(dataset, dict):
        dataset = dataset["test"]
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) if n != 'img_list' else (n, o) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    dataset = tokenize(dataset)
    print('Finished convert tokens to ids...')
    return dataset


def pad_dataset(dataset, logger, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    logger.info(f'The max length is {max_l}')
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def build_input_from_segments(persona, history, reply, img_list, tokenizer, args, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2, img_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + persona] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    input_len, persona_len = len(list(chain(*sequence))), len(persona)
    instance["input_ids"] = list(chain(*sequence)) if input_len <= args.max_length else [bos] + persona + list(chain(*sequence))[-args.max_length + 1 + persona_len:]
    instance["token_type_ids"] = [bos] + [speaker1] * persona_len + [speaker1 if i % 2 else speaker2 for i, s in enumerate(sequence[1:]) for _ in s][-args.max_length + 1 + persona_len:]
    if input_len > args.max_length:
        instance["input_ids"][1 + persona_len] = instance["token_type_ids"][1 + persona_len]  # Added 'speaker1' or 'speaker2' based on token_type_ids
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    lm_labels_ids = [-100] * len(instance["input_ids"])
    instance["lm_labels"] = lm_labels_ids[-args.max_length:]
    if lm_labels:
        lm_labels_ids = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        instance["lm_labels"] = lm_labels_ids[-args.max_length:]
    # imges
    def get_image_chars_indexes(array, token):
        """获取图片在input ids 中位置"""
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        # 查找图片字符在上下文中的位置
        indexes = np.argwhere(array == token)
        return indexes.reshape(1, -1).tolist()[0]

    images_name = [os.path.join(args.image_path, image_name) for image_name in img_list]
    images_id = get_image_chars_indexes(instance["input_ids"], img_id)
    if not images_id and len(images_id) == 0:
        images_name = []
    else:
        images_name = images_name[-len(images_id):]  # 截取input ids还存在images
    instance["image_names"] = images_name
    instance["image_ids"] = images_id
    assert len(instance["image_names"]) == len(instance["image_ids"])
    return instance


def build_dataloader(args, tokenizer, logger):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, logger)
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "dev": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0: # and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in tqdm(dataset):
            persona = dialog["personality"].copy()
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]    # +1 as question
                img_list = utterance["img_list"]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_from_segments(persona, history, candidate, img_list, tokenizer, args, lm_labels)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                datasets[dataset_name]["n_candidates"] = num_candidates

    logger.info("Pad inputs and convert to Tensor")
    data = {}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, logger, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        data[dataset_name] = dataset
    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = DialoImageDataset(data["train"], args.images_feature_path, "train"), DialoImageDataset(data["dev"], args.images_feature_path, "dev")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              collate_fn=train_dataset.collate_fn,
                              num_workers=args.num_workers,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              collate_fn=valid_dataset.collate_fn,
                              num_workers=args.num_workers,
                              shuffle=False)
    logger.info("Train dataset (Batch, Seq length): {}".format(np.array(train_dataset.dataset["input_ids"]).shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(np.array(valid_dataset.dataset["input_ids"]).shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def build_test_dataloader(args, tokenizer, logger):
    dataset = get_test_data(args.test_data_file, tokenizer)
    datasets = {"test": defaultdict(list)}
    for dialog in tqdm(dataset):
        persona = dialog["personality"].copy()
        for utterance in dialog["utterances"]:
            history = utterance["history"]
            img_list = utterance["img_list"]
            reply = []
            instance = build_input_from_segments(persona, history, reply, img_list, tokenizer, args, with_eos=False)
            for input_name, input_array in instance.items():
                datasets["test"][input_name].append(input_array)
    logger.info("Pad inputs and convert to Tensor")
    data = {}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, logger, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        data[dataset_name] = dataset
    logger.info("Build train and validation dataloaders")
    test_dataset = DialoImageDataset(data["test"], images_feature_path=None, data_type="val")
    test_loader = DataLoader(test_dataset,
                             sampler=None,
                             batch_size=1,
                             collate_fn=test_dataset.collate_fn,
                             num_workers=0,
                             shuffle=False)
    return test_loader


def get_dataloader(args, tokenizer):
    dataset = get_test_data(args.test_data_file, tokenizer)
    test_dataset = DialoDataset(dataset, tokenizer, args)
    test_loader = DataLoader(test_dataset,
                             collate_fn=test_dataset.collate,
                             pin_memory=(args.device == "cuda"),
                             num_workers=0,
                             batch_size=1,
                             shuffle=False)
    return test_loader