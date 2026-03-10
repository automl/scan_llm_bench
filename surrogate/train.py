import argparse
import json
import random

import numpy as np
import torch

from surrogate.data import SurrogateDataset
from surrogate.models.autogluon import AutoGluonModel
from surrogate.models.base import MultiLabelSurrogateModel
from surrogate.models.ensembles import BaggingEnsemble, EnsembleType
from surrogate.models.pfn import TabPFNModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv", default="splits/train.csv")
    p.add_argument("--test_csv", default="splits/test.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels", nargs="+", default=["val_loss", "test_loss"])
    p.add_argument("--model", choices=["tabpfn", "ensemble", "autogluon"], default="tabpfn")
    p.add_argument("--ensemble_type", choices=["xgb", "lightgbm", "mix"], default="mix")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results.json")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    dataset = SurrogateDataset(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        targets=args.labels,
        seed=args.seed,
    )

    sizes = dataset.get_default_sizes()
    labels = args.labels

    if args.model == "tabpfn":
        model = MultiLabelSurrogateModel(
            labels=labels,
            model_factory=lambda label: TabPFNModel(seed=args.seed, device=args.device),
        )
        model_name = "tabpfn"

    elif args.model == "ensemble":
        model = MultiLabelSurrogateModel(
            labels=args.labels,
            model_factory=lambda label: BaggingEnsemble(
                EnsembleType(args.ensemble_type),
                seed=args.seed,
            ),
        )
        model_name = f"ensemble_{args.ensemble_type}"

    elif args.model == "autogluon":

        model = MultiLabelSurrogateModel(
            labels=args.labels,
            model_factory=lambda label: AutoGluonModel(
                features=dataset.features,
                label=label,
                time_limit=200,
                base_path=f"AutogluonModels/{label}"
            ),
        )
        model_name = "autogluon"

    else:
        raise ValueError(f"Unknown model: {args.model}")

    results = model.validate(dataset, sizes)

    payload = {
        "model": model_name,
        "seed": args.seed,
        "results": results,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=4)