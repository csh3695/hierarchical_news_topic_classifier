import torch
import torch.nn as nn
from transformers import ElectraModel
from sklearn.metrics import roc_auc_score


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_labels, dropout=0.0):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Linear(in_features, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.classifier(self.dropout(x))


class TopicClassifier(nn.Module):
    def __init__(self, config):
        super(TopicClassifier, self).__init__()
        self.electra = ElectraModel.from_pretrained(
            config.electra_path,
            revision=config.electra_revision,
            use_auth_token=config.electra_use_auth_token,
        )
        self.classifier = ClassificationHead(
            self.electra.config.hidden_size,
            config.num_class,
            config.dropout,
        )

    def forward(self, data, evaluate=False, output_attentions=False):
        hidden_states = self.electra(
            input_ids=data["input_ids"],
            token_type_ids=data["token_type_ids"],
            attention_mask=data["attention_mask"],
            output_attentions=output_attentions,
        )[:]
        pooled_output = hidden_states[0][:, 0]
        logit = self.classifier(pooled_output)

        output = {"logit": logit}
        if output_attentions:
            output["attentions"] = hidden_states[1]

        if "label" in data:
            loss_per_class = nn.functional.binary_cross_entropy_with_logits(
                logit, data["label"], data["weight"], reduction="none"
            )
            # loss_per_item = loss_per_class.sum(-1) / data["weight"].sum(-1).clip(min=1)
            loss = loss_per_class.sum() / data["weight"].sum().clip(min=1)
            output.update(
                {
                    "loss": loss,
                    # "loss_per_item": loss_per_item,
                    # "loss_per_class": loss_per_class,
                }
            )

        if evaluate:
            sampled_labels = data["label"][data["weight"] == 1].detach()
            sampled_logits = logit[data["weight"] == 1].detach()
            sampled_preds = (sampled_logits > 0).float()
            auc = roc_auc_score(sampled_labels.cpu(), sampled_logits.cpu())
            output["auc"] = torch.tensor(auc).to(logit)
            output["accuracy"] = (sampled_labels == sampled_preds).float().mean()
            output["precision"] = (
                (sampled_labels == sampled_preds)[sampled_preds == 1].float().mean()
            )
            output["recall"] = (sampled_labels == sampled_preds)[
                sampled_labels == 1
            ].float()
            output["f1"] = (
                2
                * output["precision"]
                * output["recall"]
                / (output["recall"] + output["precision"]).clip(min=1e-12)
            )

            output["TP"] = (sampled_labels * sampled_preds).sum()
            output["TN"] = ((1 - sampled_labels) * (1 - sampled_preds)).sum()
            output["FP"] = ((1 - sampled_labels) * sampled_preds).sum()
            output["FN"] = (sampled_labels * (1 - sampled_preds)).sum()

        return output
