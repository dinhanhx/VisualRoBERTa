import random
import itertools
import logging
import json
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize

from tqdm import tqdm

from transformers.tokenization_utils_base import BatchEncoding

from src.tokenization import BunTokenizer


class ImageTextPair(Dataset):
    """This Dataset is purely loading collection of image text pairs
        with COCO 2017 format
        without task specific
    """

    def __init__(self, img_root_dir: Path, json_dir: Path,
                 split: str = 'train', do_sort: bool = False):
        """
        Parameters
        ----------
        img_root_dir : Path
            Directory where contains train2017/ and val2017/
        json_dir : Path
            Directory where contains COCO anntotations like json files
        split : str
            'train' or 'val'
        """
        self.img_root_dir = img_root_dir
        self.json_dir = json_dir

        self.img_split_dir = img_root_dir.joinpath(
            'train2017' if split == 'train' else 'val2017')
        self.json_split_file = json_dir.joinpath(
            'captions_train2017_trans_plus.json' if split == 'train' else 'captions_val2017_trans.json')

        self.dataset = json.load(open(self.json_split_file))['annotations']
        if do_sort:
            self.dataset.sort(key=lambda d: d['image_id'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            index in the annotation file

        Returns
        -------
        Dict
            Example return:
            {'caption': 'Một chiếc xe máy Honda màu đen đậu trước gara.',
            'id': 38,
            'image_id': 179765,
            'image_file': PosixPath('/mnt/disks/storage/val2017/000000179765.jpg')}
        """
        data = self.dataset[index]
        img_file = self.img_split_dir.joinpath(str(data['image_id']).zfill(12)+'.jpg')
        assert img_file.is_file()
        data['img_file'] = img_file
        return data

    def run_sanity_check(self):
        """Check files, directories, and images path exist or not
        """
        for data in tqdm(self.dataset):
            img_file = self.img_split_dir.joinpath(str(data['image_id']).zfill(12)+'.jpg')
            if not img_file.is_file():
                logging.warning(f'{data} @ {self.img_split_dir} has no image')


class PretrainTask(ImageTextPair):

    def __init__(self, img_root_dir: Path, json_dir: Path, split: str = 'train', do_sort: bool = False):
        super().__init__(img_root_dir, json_dir, split, do_sort)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index: int):
        true_pair = super().__getitem__(index)

        # Get a data point randomly then replace it's `img_file` with true_pair's
        # to make a mismatch data
        random_text = random.choice(self.dataset)
        while random_text['image_id'] == true_pair['image_id']:
            random_text = random.choice(self.dataset)

        random_text['img_file'] = true_pair['img_file']
        return true_pair, random_text


@dataclass
class PretrainCollator:
    tokenizer: BunTokenizer
    image_size: list
    patch_size: list

    def __call__(self, batch_inputs):
        # Create labels to indicate which is true pair, which is not
        # 0 - mismatch, 1 - true pair
        match_labels = torch.ones(len(batch_inputs)*2).long()
        match_labels[1::2] = 0

        batch_inputs = itertools.chain(*batch_inputs)
        batch_texts = []
        batch_imgs = []
        for i in batch_inputs:
            batch_texts.append(i['caption'])
            batch_imgs.append(i['img_file'])

        text_inputs = self.tokenizer(batch_texts, return_tensors='pt', padding='max_length', max_length=85)
        masked_text_inputs = self.mask_whole_work(text_inputs)
        image_inputs = self.tensorize_image_batch(batch_imgs)

        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        # Extend the shape of text_inputs.attention_masks to cover image_inputs
        extra_attention_mask = torch.ones(match_labels.shape[0], num_patches, dtype=text_inputs.attention_mask.dtype)
        attention_mask = torch.cat((text_inputs.attention_mask, extra_attention_mask), dim=1)

        # Extend the shape of masked_text_inputs['labels'] to cover image_inputs
        extra_labels = torch.full((match_labels.shape[0], num_patches), -100)
        labels = torch.cat((masked_text_inputs['labels'], extra_labels), dim=1)

        return BatchEncoding({'input_ids': masked_text_inputs['input_ids'],
                              'labels': labels,
                              'match_labels': match_labels,
                              'attention_mask': attention_mask,
                              'token_type_ids': text_inputs.token_type_ids,
                              'image_input': image_inputs})

    def tensorize_image_batch(self, batch_imgs):
        return torch.stack([resize(read_image(str(img), ImageReadMode.RGB),
                                   self.image_size) for img in batch_imgs],
                           0).float()

    def mask_whole_work(self, text_inputs: BatchEncoding):
        seq_ids = torch.clone(text_inputs.input_ids)
        labels = torch.full(seq_ids.shape, -100)
        for seq_id, label in zip(seq_ids, labels):
            tokens = [self.tokenizer._convert_id_to_token(id.item()) for id in seq_id]  # type: ignore

            cand_indices = []
            for i, token in enumerate(tokens):
                if token == '[CLS]' or token == '[SEP]':
                    continue

                if len(cand_indices) >= 1 and not token.startswith('▁'):
                    cand_indices[-1].append(i)
                else:
                    cand_indices.append([i])
            to_mask_index = random.choice(cand_indices)
            label[to_mask_index] = seq_id[to_mask_index]
            seq_id[to_mask_index] = self.tokenizer.mask_token_id  # type: ignore

        return {'input_ids': seq_ids,
                'labels': labels}


class VisualQuestionAnswer(Dataset):
    """This Dataset purely load collection of image, question, answer triplets
        with ViVQA format
    """

    def __init__(self, img_root_dir: Path, csv_dir: Path,
                 split: str = 'train'):
        """
        Parameters
        ----------
        img_root_dir : Path
            Directory where contains train2017/ and val2017/
        csv_dir : Path
            Directory where contains all csv files
            Where github.com/kh4nh12/ViVQA is cloned
        split : str, optional
            'train' or 'test', by default 'train'
        """
        self.img_root_dir = img_root_dir
        self.img_train_dir = img_root_dir.joinpath('train2017')
        self.img_val_dir = img_root_dir.joinpath('val2017')

        self.csv_split_file = csv_dir.joinpath(f'{split}.csv')

        self.dataset = pd.read_csv(self.csv_split_file, index_col=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            index in the csv file

        Returns
        -------
        Dict
            Example return:
            {'img_file': PosixPath('/mnt/disks/nlpvnhub/dinhanhx/train2017/000000131731.jpg'),
            'question': 'màu của lông là gì',
            'answer': 'màu đen'}
        """
        data = self.dataset.iloc[index]
        for img_dir in [self.img_train_dir, self.img_val_dir]:
            img_file = img_dir.joinpath(str(data['img_id']).zfill(12)+'.jpg')
            if img_file.is_file():
                return {'img_file': img_file,
                        'question': data['question'],
                        'answer': data['answer']}

    def run_sanity_check(self):
        """Check files, directories, and images path exist or not
        """
        for index, data in tqdm(self.dataset.iterrows()):
            file_found = False
            for img_dir in [self.img_train_dir, self.img_val_dir]:
                img_file = img_dir.joinpath(str(data['img_id']).zfill(12)+'.jpg')
                if img_file.is_file():
                    file_found = True

            if not file_found:
                logging.warn(f'{data} @ {self.csv_split_file} has no image')


@dataclass
class VisualQuestionAnswerCollator:
    tokenizer: BunTokenizer
    image_size: list
    patch_size: list

    def __call__(self, batch_inputs):
        batch_length = len(batch_inputs)
        batch_texts = []
        batch_imgs = []
        for i in batch_inputs:
            batch_texts.append(i['question'] + ' ? ' + i['answer'])
            batch_imgs.append(i['img_file'])

        text_inputs = self.tokenizer(batch_texts,
                                     return_tensors='pt',
                                     padding='max_length',
                                     max_length=37)
        labels = self.make_labels(text_inputs.input_ids)
        image_inputs = self.tensorize_image_batch(batch_imgs)

        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        # Extend the shape of text_inputs.attention_masks to cover image_inputs
        extra_attention_mask = torch.ones(batch_length, num_patches, dtype=text_inputs.attention_mask.dtype)
        attention_mask = torch.cat((text_inputs.attention_mask, extra_attention_mask), dim=1)

        # Extend the shape of labels to cover image_inputs
        extra_labels = torch.full((batch_length, num_patches), -100)
        labels = torch.cat((labels, extra_labels), dim=1)

        return BatchEncoding({'input_ids': text_inputs.input_ids,
                              'labels': labels,
                              'attention_mask': attention_mask,
                              'token_type_ids': text_inputs.token_type_ids,
                              'image_input': image_inputs})

    def tensorize_image_batch(self, batch_imgs):
        return torch.stack([resize(read_image(str(img), ImageReadMode.RGB),
                                   self.image_size) for img in batch_imgs],
                           0).float()

    def make_labels(self, input_ids):
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return labels


@dataclass
class ImageCaptioningCollator:
    tokenizer: BunTokenizer
    image_size: list
    patch_size: list

    def __call__(self, batch_inputs):
        batch_length = len(batch_inputs)
        batch_texts = []
        batch_imgs = []
        for i in batch_inputs:
            batch_texts.append(i['caption'])
            batch_imgs.append(i['img_file'])

        text_inputs = self.tokenizer(batch_texts,
                                     return_tensors='pt',
                                     padding='max_length',
                                     max_length=85)
        labels = self.make_labels(text_inputs.input_ids)
        image_inputs = self.tensorize_image_batch(batch_imgs)

        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        # Extend the shape of text_inputs.attention_masks to cover image_inputs
        extra_attention_mask = torch.ones(batch_length, num_patches, dtype=text_inputs.attention_mask.dtype)
        attention_mask = torch.cat((text_inputs.attention_mask, extra_attention_mask), dim=1)

        # Extend the shape of labels to cover image_inputs
        extra_labels = torch.full((batch_length, num_patches), -100)
        labels = torch.cat((labels, extra_labels), dim=1)

        return BatchEncoding({'input_ids': text_inputs.input_ids,
                              'labels': labels,
                              'attention_mask': attention_mask,
                              'token_type_ids': text_inputs.token_type_ids,
                              'image_input': image_inputs})

    def tensorize_image_batch(self, batch_imgs):
        return torch.stack([resize(read_image(str(img), ImageReadMode.RGB),
                                   self.image_size) for img in batch_imgs],
                           0).float()

    def make_labels(self, input_ids):
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return labels
