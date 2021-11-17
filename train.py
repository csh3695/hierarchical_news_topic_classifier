import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import collate_fn
from dataset import HierarchicalTopicDataset
from model import TopicClassifier
from optimizer import get_optimizer
from utils import TopicClassifierArgParser
import warnings
import logging

if __name__ == "__main__":
    problem_logger = logging.getLogger('transformers.tokenization_utils_base')
    problem_logger.setLevel("ERROR")
    argparser = TopicClassifierArgParser()
    args, config = argparser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, args.cuda_id)))

    dataset = HierarchicalTopicDataset(config)

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn(verbose=False),
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=True,
    )
    print("# Data:", len(dataset))
    print("# Steps per Epoch:", len(dataloader))

    model = TopicClassifier(config)
    model.cuda()
    model = nn.DataParallel(model)
    optimizer, scheduler = get_optimizer(model, config, config.num_train_steps)

    model.train()

    runpath = Path("./experiment") / args.run_name
    runpath.mkdir(parents=True, exist_ok=True)
    with open(runpath / "config.json", "w") as f:
        json.dump(config.toDict(), f)

    wandb.init(project="TopicClassifier", config=config, entity='csh3695')
    wandb.run.name = args.run_name
    train_step = 0
    for ep in range(config.num_train_steps // len(dataloader) + 1):
        tdl = tqdm(dataloader)
        for i, data in enumerate(tdl):
            train_step += 1
            do_logging = (train_step - 1) * (train_step % config.logging_interval) == 0
            do_saving = (
                (train_step - 1) * (train_step % config.saving_interval) == 0
            ) or (train_step == config.num_train_steps)

            optimizer.zero_grad()
            data = {
                k: data[k].cuda() if isinstance(data[k], torch.Tensor) else data[k]
                for k in data
            }
            out = model(data, evaluate=(do_logging or do_saving))
            loss = out["loss"].mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tdl.set_description(f"EP {ep:2d}| Loss {loss.item():.4f}", refresh=False)

            if do_logging or do_saving:
                logging_summary = {"step": train_step, "train_loss": loss.item()}
                logging_summary.update(
                    {k: out[k].mean().item() for k in out if k != "logit"}
                )
                wandb.log(logging_summary)

                if do_saving:
                    svdir = runpath / str(train_step)
                    svdir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), svdir / f"model.pth")
                    torch.save(optimizer.state_dict(), svdir / f"optim.pth")
                    torch.save(scheduler.state_dict(), svdir / f"scheduler.pth")

                if train_step == config.num_train_steps:
                    break

from transformers import ElectraTokenizer, ElectraModel
ElectraTokenizer.from_pretrained('ryanc/knelectra-small-discriminator', use_auth_token=True)
ElectraModel.from_pretrained('ryanc/knelectra-small-discriminator', use_auth_token=True)