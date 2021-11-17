import json
from copy import deepcopy

import torch
import torch.nn as nn
from dotmap import DotMap
from transformers import ElectraTokenizer

from model import TopicClassifier
from preprocess import Preprocessor

from collections import namedtuple

summary = namedtuple(
    "Predictions", ("prob", "pred_by_threshold", "pred_by_greedy", "attention")
)


class TopicPredictor(nn.Module):
    def __init__(self, run_path, ep, cuda=True):
        super(TopicPredictor, self).__init__()
        self.run_path = run_path
        self.ep = ep
        self.cuda = cuda
        self.config = self.load_config()
        self.model = self.load_model()
        self.tokenizer = ElectraTokenizer.from_pretrained(
            self.config.electra_path,
            revision=self.config.electra_revision,
            use_auth_token=self.config.electra_use_auth_token,
        )
        self.category = self.load_category(self.config.category_path)

    def load_config(self):
        with open(f"{self.run_path}/config.json", "r") as f:
            return DotMap(json.load(f))

    def load_model(self):
        model = TopicClassifier(self.config)
        state_dict = torch.load(f"{self.run_path}/{self.ep}/model.pth", map_location=('cuda' if self.cuda else 'cpu'))
        state_dict = {
            k[7:] if k.startswith("module") else k: state_dict[k] for k in state_dict
        }
        model.load_state_dict(state_dict)
        if self.cuda:
            model = model.cuda()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def load_category(category_path):
        with open(category_path, "r") as f:
            return json.load(f)

    @staticmethod
    def dfs(root, bucket):
        for k in root:
            bucket.append(k)
            if root[k][0]:
                TopicPredictor.dfs(root[k][0], bucket)

    @staticmethod
    def get_category_by_index(root, index, getter=lambda key, root: key):
        for k in root:
            if root[k][2] == index:
                return [getter(k, root)]
            else:
                result = TopicPredictor.get_category_by_index(root[k][0], index, getter)
                if result:
                    return [getter(k, root)] + result
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
            _p, _n = TopicPredictor.get_sample_index(root[k][0], target[k])
            positive_sample_index.extend(_p)
            negative_sample_index.extend(_n)

        return positive_sample_index, negative_sample_index

    @staticmethod
    def get_category_by_condition(
        root, condition=lambda key, root: True, getter=lambda key, root: key
    ):
        result = []
        for k in root:
            if condition(k, root):
                child_res = TopicPredictor.get_category_by_condition(
                    root[k][0], condition, getter
                )
                if child_res:
                    result.extend(
                        list(
                            map(lambda x: [getter(k, root)] + x, child_res),
                        )
                    )
                else:
                    result.append([getter(k, root)])
        return result

    def decoder(self, logit):
        cate = deepcopy(self.category)

        def _assign_conditional_prob(_c):
            for k in _c:
                _c[k].append(logit[_c[k][2]])
                if _c[k][0]:
                    _assign_conditional_prob(_c[k][0])

        def _calculate_total_prob(_c, prior):
            for k in _c:
                _c[k].append(_c[k][3] * prior)
                if _c[k][0]:
                    _calculate_total_prob(_c[k][0], _c[k][4])

        _assign_conditional_prob(cate)
        _calculate_total_prob(cate, 1.0)
        return cate

    def preprocess(self, title, content):
        if isinstance(title, str):
            title = [title]
        if isinstance(content, str):
            content = [content]
        title = list(
            map(lambda x: Preprocessor.preprocess(x, to_lower=True, split=False), title)
        )
        content = list(
            map(
                lambda x: Preprocessor.preprocess(x, to_lower=True, split=False),
                content,
            )
        )
        encoded = self.tokenizer.batch_encode_plus(
            list(zip(title, content)),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors='pt',
            verbose=False
        )
        if self.cuda:
            encoded = dict((k, encoded[k].cuda()) for k in encoded)
        return encoded

    def greedy_max_prob_find(
        self,
        root,
        condition=lambda root, prev, cur: root[prev][-2] < root[cur][-2],
        getter=lambda key, root: key,
    ):
        selected = None
        for key in root:
            if not selected:
                selected = key
            else:
                selected = key if condition(root, selected, key) else selected
        if selected is None:
            return []
        return [getter(selected, root)] + self.greedy_max_prob_find(
            root[selected][0], condition, getter
        )

    def postprocess(self, model_input, model_output, threshold=None, greedy=False):
        def _getter(key, root):
            return {
                "topic_index": root[key][2],
                "topic": key,
                "cond_prob": root[key][-2],
                "prob": root[key][-1],
            }  # root[key] = [..., cond_prob, total_prob]

        logits = model_output["logit"]

        prob_result = []
        pred_by_threshold_result = []
        pred_by_greedy_result = []
        for logit in logits:
            cate_probs = self.decoder(logit.sigmoid().tolist())
            probs = [
                self.get_category_by_index(
                    cate_probs,
                    x,
                    getter=_getter,
                )
                for x in range(self.config.num_class)
            ]
            prob_result.append(sorted(probs, key=lambda x: x[-1]["prob"], reverse=True))
            if greedy:
                pred_by_greedy_result.append(
                    self.greedy_max_prob_find(
                        cate_probs,
                        condition=lambda root, prev, cur: root[prev][-2]
                        < root[cur][-2],
                        getter=lambda key, root: key,
                    )
                )
            if threshold:
                pred_by_threshold_result.append(
                    self.get_category_by_condition(
                        cate_probs,
                        condition=lambda key, root: root[key][-2] >= threshold,
                        getter=lambda key, root: key,
                    )
                )

        calc_attentions = None
        if "attentions" in model_output:
            attentions = torch.stack(
                model_output["attentions"], 1
            )  # Batch x Layer x Head x SeqLen x SeqLen
            cls_attentions = attentions[:, :, :, 0, :]  # Batch x Layer x Head x SeqLen
            calc_attentions = self.calc_attentions(
                model_input["input_ids"], model_input["attention_mask"], cls_attentions
            )

        return summary(
            prob=prob_result,
            pred_by_threshold=pred_by_threshold_result,
            pred_by_greedy=pred_by_greedy_result,
            attention=calc_attentions,
        )

    def forward(
        self, title, content, threshold=None, output_attentions=False, greedy=False
    ):
        batch = self.preprocess(title, content)
        out = self.model(batch, output_attentions=output_attentions)
        summary = self.postprocess(batch, out, threshold, greedy)
        return summary

    def calc_attentions(
        self, batched_input_ids, batched_attention_mask, batched_attentions
    ):
        results = []
        for input_ids, attention_mask, attentions in zip(
            batched_input_ids, batched_attention_mask, batched_attentions
        ):
            attention_mask = attention_mask.bool()
            tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids.masked_select(attention_mask), skip_special_tokens=False
            )
            attention_at_token = attentions.permute(2, 0, 1)[attention_mask]
            results.append(list(zip(tokens, attention_at_token)))

        return results
