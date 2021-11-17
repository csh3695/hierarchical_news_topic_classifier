import json

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import ElectraTokenizer
from preprocess import Preprocessor


class HierarchicalTopicDataset(Dataset):
    def __init__(self, config, **kwargs):
        config.update(kwargs)
        self.config = config
        self.tokenizer = ElectraTokenizer.from_pretrained(
            config.electra_path,
            revision=config.electra_revision,
            use_auth_token=config.electra_use_auth_token,
        )
        self.article = self.load_article(config.article_path)
        self.category = self.load_category(config.category_path)
        self._cat_flatten = []
        self.dfs(self.category, self._cat_flatten)

    def _load_article(self, article_path):
        with open(article_path, "r") as f:
            article = json.load(f)
        article = pd.DataFrame.from_records(article)
        return article

    def load_article(self, article_path):
        if isinstance(article_path, list):
            article = pd.concat(
                [self._load_article(p) for p in tqdm(article_path)]
            ).reset_index(drop=True)
        elif isinstance(article_path, str):
            article = self._load_article(article_path)
        else:
            raise ValueError(f"Type {type(article_path)} for path not supported.")
        return article

    def load_category(self, category_path):
        with open(category_path, "r") as f:
            return json.load(f)

    @staticmethod
    def dfs(root, bucket):
        for k in root:
            bucket.append(k)
            if root[k][0]:
                HierarchicalTopicDataset.dfs(root[k][0], bucket)

    @staticmethod
    def get_category_by_index(root, index):
        for k in root:
            if root[k][2] == index:
                return [k]
            else:
                result = HierarchicalTopicDataset.get_category_by_index(
                    root[k][0], index
                )
                if result:
                    return [k] + result
        return None

    @staticmethod
    def get_sample_index(root, target):
        positive_sample_index = []
        negative_sample_index = []
        key = set(target.keys())
        for k in root:
            if k in key:
                positive_sample_index.append(root[k][2])
            else:
                negative_sample_index.append(root[k][2])

        for k in target:
            _p, _n = HierarchicalTopicDataset.get_sample_index(root[k][0], target[k])
            positive_sample_index.extend(_p)
            negative_sample_index.extend(_n)

        return positive_sample_index, negative_sample_index

    def __getitem__(self, item):
        target = self.article.loc[item]
        title = Preprocessor.preprocess(target.title, to_lower=True, split=False)
        content = Preprocessor.preprocess(
            " ".join(target.content), to_lower=True, split=False
        )
        encoded = self.tokenizer.encode_plus(
            title, content, max_length=self.config.max_seq_length, verbose=False
        )
        pos_idx, neg_idx = self.get_sample_index(self.category, target.topic)
        label = torch.zeros(len(self._cat_flatten), dtype=torch.float)
        label[pos_idx] = 1
        weight = torch.zeros(len(self._cat_flatten), dtype=torch.float)
        weight[pos_idx + neg_idx] = 1
        return {
            "id": item,
            "input_ids": encoded["input_ids"],
            "token_type_ids": encoded["token_type_ids"],
            "attention_mask": encoded["attention_mask"],
            "topic": target.topic,
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
            "label": label,
            "weight": weight,
        }

    def __len__(self):
        return len(self.article)
