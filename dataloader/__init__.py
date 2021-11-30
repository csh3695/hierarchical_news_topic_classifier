import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(verbose: bool):
    def _collate_fn(samples):
        batch = dict()
        if verbose:
            for key in ["id", "topic", "pos_idx", "neg_idx"]:
                batch[key] = [sample[key] for sample in samples]
        for key in ["input_ids", "token_type_ids", "attention_mask"]:
            batch[key] = pad_sequence(
                [torch.tensor(sample[key], dtype=torch.long) for sample in samples],
                batch_first=True,
                padding_value=0,
            )
        for key in ["label", "weight"]:
            batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

        return batch

    return _collate_fn
