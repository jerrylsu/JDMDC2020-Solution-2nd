from itertools import chain
import torch
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torchvision import transforms
import PIL
import numpy as np

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]", "<img>", "[PAD]"]

# temporarily use resent18 image statistics
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class DialoDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        persona = self.data[item]['personality']
        history = self.data[item]['utterances'][0]['history']
        img_list = self.data[item]['utterances'][0]['img_list']
        reply = []
        return self.process(persona, history, reply, img_list, self.tokenizer, self.args, with_eos=False)

    def process(self, persona, history, reply, img_list, tokenizer, args, lm_labels=False, with_eos=True):
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
        def image_transform(images, data_type):
            """Read all image data in the single utterance."""
            resp_list = []
            for image in images:
                img = torch.zeros(3, 224, 224)
                try:
                    img_tmp = PIL.Image.open(image)
                    img = data_transforms[data_type](img_tmp)
                except:
                    print("can't open image file: ", image)
                    pass
                finally:
                    resp_list.append(img)
            return resp_list  # 没有图片直接传空list
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
        instance["input_images"] = image_transform(images_name, "val")
        instance["image_ids"] = images_id
        assert len(instance["input_images"]) == len(instance["image_ids"])
        return instance["input_ids"], instance["token_type_ids"], instance["input_images"], instance["image_ids"]

    def collate(self, batch):
        # input_ids = pad_sequence(
        #     [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
        #     batch_first=True, padding_value=self.pad)
        # token_type_ids = pad_sequence(
        #     [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
        #     batch_first=True, padding_value=self.pad)
        # image_names, image_ids = (batch[0]["image_names"],), (batch[0]["image_ids"],)
        # return input_ids, token_type_ids, image_names, image_ids
        input_ids, token_type_ids, input_images, image_ids = zip(*batch)
        return input_ids, token_type_ids, input_images, image_ids


class DialoImageDataset(Dataset):
    def __init__(self, dataset, images_feature_path, data_type):
        self.dataset = dataset
        self.data_type = data_type
        self.images_feature_json = torch.load(images_feature_path) if images_feature_path else None

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, item):
        input_ids = self.dataset["input_ids"][item]
        token_type_ids = self.dataset["token_type_ids"][item]
        if self.images_feature_json:
            input_images = self.get_images_feature(self.dataset["image_names"][item])
        else:
            input_images = self.image_transform(self.dataset["image_names"][item], self.data_type)
        image_ids = self.dataset["image_ids"][item]
        lm_labels = self.dataset["lm_labels"][item]
        mc_token_ids = self.dataset["mc_token_ids"][item]
        mc_labels = self.dataset["mc_labels"][item]
        return input_ids, token_type_ids, input_images, image_ids, lm_labels, mc_token_ids, mc_labels

    def image_transform(self, images, data_type):
        """Read all image data in the single utterance."""
        resp_list = []
        for image in images:
            img = torch.zeros(3, 224, 224)
            try:
                img_tmp = PIL.Image.open(image)
                img = data_transforms[data_type](img_tmp)
            except:
                print("can't open image file: ", image)
                pass
            finally:
                resp_list.append(img)
        return resp_list   # 没有图片直接传空list

    def get_images_feature(self, images_name):
        """获取image feature"""
        images_feature = []
        for image_name in images_name:
            tmp = self.images_feature_json.get(image_name, np.zeros(512, dtype=np.float32))
            images_feature.append(torch.from_numpy(tmp))

        return images_feature

    def get_images_feature_padding(self, single_images_name, single_images_id, sentence_length):
        """获取image feature
        @param single_images_name:  images name(key)
        @param single_images_id:
        @param sentence_length:
        @return:
        """
        sample_image_embed = np.zeros(512)  # image feature sample
        sentence_embeds = [sample_image_embed] * sentence_length
        assert len(single_images_name) == len(single_images_id)
        if single_images_name and len(single_images_name) > 0:
            for idx, image_name in enumerate(single_images_name):
                i = single_images_id[idx]
                image_feature = self.images_feature_json.get(image_name)
                sentence_embeds[i] = image_feature
        return sentence_embeds

    def collate_fn(self, batch):
        input_ids, token_type_ids, input_images, image_ids, lm_labels, mc_token_ids, mc_labels = zip(*batch)
        return input_ids, token_type_ids, input_images, image_ids, lm_labels, mc_token_ids, mc_labels
