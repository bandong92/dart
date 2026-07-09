import argparse
import json
import os
import sys

from training import run_export_onnx_job, run_test_job, run_training_job


def build_parser():
    parser = argparse.ArgumentParser(
        description="DART CLI for self-supervised STN training, testing, and ONNX export."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("train", "test", "export-onnx"),
        help="Execution mode.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file. Command-line arguments override file values.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Dataset folder containing cad and ori subfolders.",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Path to a trained checkpoint file.",
    )
    parser.add_argument(
        "--resume-from",
        help="Checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--onnx-output-path",
        help="Output path for exported ONNX file.",
    )
    parser.add_argument("--train-ratio", type=float, help="Train split ratio or percent value.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        help="Validation split ratio or percent value.",
    )
    parser.add_argument("--test-ratio", type=float, help="Test split ratio or percent value.")
    parser.add_argument("--split-seed", type=int, help="Random seed for dataset split.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, help="Learning rate.")
    return parser


def load_config(path):
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def merge_config(file_config, args):
    merged = dict(file_config)
    cli_values = {
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint_path,
        "resume_from": args.resume_from,
        "onnx_output_path": args.onnx_output_path,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    for key, value in cli_values.items():
        if value is not None:
            merged[key] = value
    return merged


def normalize_config(config):
    normalized = dict(config)
    for key in ("train_ratio", "validation_ratio", "test_ratio", "learning_rate"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = float(value)

    for key in ("epochs", "batch_size", "split_seed"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = int(value)

    return normalized


def require_keys(config, keys):
    missing = [key for key in keys if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")


def log_message(message):
    print(message, flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = normalize_config(merge_config(load_config(args.config), args))

    try:
        if args.mode == "train":
            require_keys(
                config,
                [
                    "dataset_path",
                    "train_ratio",
                    "validation_ratio",
                    "test_ratio",
                    "epochs",
                    "batch_size",
                    "learning_rate",
                ],
            )
            config.setdefault("split_seed", 42)
            result = run_training_job(config, progress_callback=log_message)
            print(json.dumps(result, indent=2), flush=True)
            return 0

        if args.mode == "test":
            require_keys(
                config,
                [
                    "dataset_path",
                    "checkpoint_path",
                    "train_ratio",
                    "validation_ratio",
                    "test_ratio",
                    "batch_size",
                ],
            )
            config.setdefault("split_seed", 42)
            result = run_test_job(config, progress_callback=log_message)
            print(json.dumps(result, indent=2), flush=True)
            return 0

        require_keys(config, ["checkpoint_path"])
        output_path = config.get("onnx_output_path") or os.path.join(
            os.getcwd(),
            "outputs",
            "dart_stn.onnx",
        )
        result = run_export_onnx_job(
            config["checkpoint_path"],
            output_path,
            progress_callback=log_message,
        )
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
