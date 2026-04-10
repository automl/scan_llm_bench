import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from llm_bench.data import SurrogateDataset
from llm_bench.models.autogluon import AutoGluonModel
from llm_bench.metrics import compute_regression_metrics
from llm_bench.models.base import MultiLabelSurrogateModel
from llm_bench.models.ensembles import BaggingEnsemble, EnsembleType
from llm_bench.models.pfn import TabPFNModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv", default="new_surrogate_data/train.csv")
    p.add_argument("--test_csv", default="new_surrogate_data/test.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels", nargs="+", default=["valid_loss","test_loss",
                                                   # "hellaswag_acc",
                                                   # "arc_challenge_acc", "arc_easy_acc", "copa_acc", "openbookqa_acc",
                                                   # "piqa_acc"
                                                   ])
    p.add_argument("--model", choices=["tabpfn", "ensemble", "autogluon"], default="tabpfn")
    p.add_argument("--ensemble_type", choices=["xgb", "lightgbm", "mix"], default="lightgbm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/seed_42/full_rows/new_surrogate_data_full_rows_tabpfn_with_uncertainty.json")

    return p.parse_args()


def _evaluate_with_uncertainty(model: MultiLabelSurrogateModel, dataset: SurrogateDataset, sizes):
    X_test, y_test = dataset.get_test_data()
    results = {}

    for n_cfg in sizes:
        X_sub, y_sub = dataset.get_train_subset(n_cfg)
        per_label_metrics = {}

        for i, label in enumerate(model.labels):
            predictor = model.get_predictor(label)
            predictor.fit(X_sub, y_sub[:, i])

            outputs = predictor.predict_with_uncertainty(X_test)
            mean_pred = np.asarray(outputs.get("mean")).reshape(-1)

            per_label_metrics[label] = compute_regression_metrics(y_test[:, i], mean_pred)
            per_label_metrics[label]["mean"] = float(np.average(mean_pred))
            per_label_metrics[label]['uncertainty'] = float(np.average(outputs["uncertainty_width"]))

        results[int(n_cfg)] = per_label_metrics

    return results


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
                time_limit=5*60,
                base_path=f"AutogluonModels/seed_{args.seed}/train_last_rows/{label}"
            ),
        )
        model_name = "autogluon"

    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.model == "tabpfn":
        results= _evaluate_with_uncertainty(model, dataset, sizes)
    else:
        results = model.validate(dataset, sizes)

    payload = {
        "model": model_name,
        "seed": args.seed,
        "results": results,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=4)