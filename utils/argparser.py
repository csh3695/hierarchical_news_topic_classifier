import argparse
from time import strftime
from dotmap import DotMap


class TopicClassifierArgParser:
    config_keys = [
        "article_path",
        "category_path",
        "electra_path",
        "electra_revision",
        "electra_use_auth_token",
        "max_seq_length",
        "num_class",
        "dropout",
        "batch_size",
        "learning_rate",
        "warmup_proportion",
        "num_train_steps",
        "num_eval_steps",
        "logging_interval",
        "saving_interval",
    ]

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Arguments Parser for Topic Classification with KoELECTRA."
        )
        self.parser.add_argument(
            "--run_name", type=str, default=strftime("%y%m%d_%H%M%S")
        )
        self.parser.add_argument("--cuda_id", type=int, nargs="+", default=[0, 1])

        self.parser.add_argument(
            "--article_path",
            type=str,
            nargs="+",
            default=["./data/raw/raw_news_dep2.json", "./data/raw/raw_news_dep3.json"],
        )
        self.parser.add_argument(
            "--category_path", type=str, default="./data/topic_tree.json"
        )
        self.parser.add_argument(
            "--electra_path",
            type=str,
            default="monologg/koelectra-small-v3-discriminator",
        )
        self.parser.add_argument(
            "--electra_revision",
            type=str,
            default="main",
        )
        self.parser.add_argument(
            "--electra_use_auth_token",
            type=str,
            choices=('True', 'False'),
            default='False',
        )
        self.parser.add_argument("--max_seq_length", type=int, default=512)
        self.parser.add_argument("--num_class", type=int, default=889)
        self.parser.add_argument("--dropout", type=float, default=0.1)
        self.parser.add_argument("--batch_size", type=int, default=160)
        self.parser.add_argument("--learning_rate", type=float, default=5e-5)
        self.parser.add_argument("--warmup_proportion", type=float, default=0.1)
        self.parser.add_argument("--num_train_steps", type=int, default=50000)
        self.parser.add_argument("--num_eval_steps", type=int, default=10)
        self.parser.add_argument("--logging_interval", type=int, default=100)
        self.parser.add_argument("--saving_interval", type=int, default=1000)

    def parse_args(self):
        args = self.parser.parse_args()
        argdict = vars(args)
        argdict['electra_use_auth_token'] = (argdict['electra_use_auth_token'] == 'True')
        return args, DotMap(dict((k, argdict[k]) for k in self.config_keys))
