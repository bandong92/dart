import json
import os
import traceback

from data_pipeline import (
    ORI_HEIGHT,
    ORI_WIDTH,
    RecursiveImageDataset,
    build_dataloader,
    collect_paired_samples,
    split_paired_samples,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.utils import save_image

    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    nn = None
    F = None
    save_image = None
    TORCH_IMPORT_ERROR = str(exc)

try:
    import onnxruntime as ort

    ONNXRUNTIME_IMPORT_ERROR = None
except Exception as exc:
    ort = None
    ONNXRUNTIME_IMPORT_ERROR = str(exc)

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal


class TrainingSignals(QObject):
    started = pyqtSignal(str)
    progress = pyqtSignal(str)
    validation_preview = pyqtSignal(dict)
    test_result_batch = pyqtSignal(object)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)


class ExportSignals(QObject):
    started = pyqtSignal(str)
    progress = pyqtSignal(str)
    onnx_result_batch = pyqtSignal(object)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)


if torch is not None:
    class STNAligner(nn.Module):
        def __init__(self):
            super().__init__()
            self.localization = nn.Sequential(
                nn.Conv2d(6, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((8, 8)),
            )
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 8 * 8, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2),
            )
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

        def estimate_theta(self, fused_images):
            xs = self.localization(fused_images)
            xs = xs.view(-1, 10 * 8 * 8)
            return self.fc_loc(xs).view(-1, 2, 3)

        def stn(self, fused_images, cad_images):
            theta = self.estimate_theta(fused_images)
            grid = F.affine_grid(theta, cad_images.size(), align_corners=False)
            return F.grid_sample(cad_images, grid, align_corners=False)

        def forward(self, fused_images, cad_images=None):
            if cad_images is None:
                cad_images = fused_images[:, :3, :, :]
            return self.stn(fused_images, cad_images)

    STNClassifier = STNAligner


class TrainingTask(QRunnable):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.signals = TrainingSignals()

    def run(self):
        if torch is None:
            self.signals.failed.emit(
                f"PyTorch is not available. Install dependencies first.\n{TORCH_IMPORT_ERROR or ''}".strip()
            )
            return

        try:
            self._run_training()
        except Exception:
            self.signals.failed.emit(traceback.format_exc())

    def _run_training(self):
        dataset_path = self.config["dataset_path"]
        train_ratio = self.config["train_ratio"]
        validation_ratio = self.config["validation_ratio"]
        test_ratio = self.config["test_ratio"]
        split_seed = self.config.get("split_seed", 42)
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]

        self.signals.started.emit("Preparing datasets...")

        all_samples = collect_paired_samples(dataset_path)
        split_samples = split_paired_samples(
            all_samples,
            train_ratio,
            validation_ratio,
            test_ratio,
            seed=split_seed,
        )

        train_dataset = RecursiveImageDataset(
            dataset_path,
            samples=split_samples["train"],
            rotation=(-10.0, 10.0),
            cache_mode="source",
        )
        validation_dataset = None
        if split_samples["validation"]:
            validation_dataset = RecursiveImageDataset(
                dataset_path,
                samples=split_samples["validation"],
                rotation=None,
                cache_mode="processed",
            )

        test_dataset = None
        if split_samples["test"]:
            test_dataset = RecursiveImageDataset(
                dataset_path,
                samples=split_samples["test"],
                rotation=None,
                cache_mode="processed",
            )

        pin_memory = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 1
        train_workers = min(12, max(0, cpu_count - 1))
        eval_workers = min(8, max(0, max(1, cpu_count // 2)))

        train_loader = build_dataloader(
            train_dataset, batch_size, True, pin_memory, num_workers=train_workers
        )
        validation_loader = None
        if validation_dataset is not None:
            validation_loader = build_dataloader(
                validation_dataset,
                batch_size,
                False,
                pin_memory,
                num_workers=eval_workers,
            )

        test_loader = None
        if test_dataset is not None:
            test_loader = build_dataloader(
                test_dataset,
                batch_size,
                False,
                pin_memory,
                num_workers=eval_workers,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = STNAligner().to(device)
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        latest_model_path = os.path.join(output_dir, "dart_stn_latest.pt")
        best_model_path = os.path.join(output_dir, "dart_stn_best.pt")
        metadata_path = os.path.join(output_dir, "dart_training_config.json")

        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(
                {
                    "dataset_path": dataset_path,
                    "train_ratio": train_ratio,
                    "validation_ratio": validation_ratio,
                    "test_ratio": test_ratio,
                    "split_seed": split_seed,
                    "split_counts": {
                        "train": len(split_samples["train"]),
                        "validation": len(split_samples["validation"]),
                        "test": len(split_samples["test"]),
                    },
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "objective": "self_supervised_cad_to_ori_alignment",
                    "loss": "SmoothL1Loss",
                },
                metadata_file,
                indent=2,
            )

        best_validation_loss = float("inf")
        history = []
        self.signals.progress.emit(
            f"Self-supervised training started on {device.type.upper()} with {len(train_dataset):,} training pairs. "
            f"train workers={train_workers}, eval workers={eval_workers}, "
            f"train cache=source, eval cache=processed."
        )

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            total = 0

            for fused_images, cad_images, ori_images in train_loader:
                fused_images = fused_images.to(device, non_blocking=pin_memory)
                cad_images = cad_images.to(device, non_blocking=pin_memory)
                ori_images = ori_images.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                aligned_cad_images = model(fused_images, cad_images)
                loss = criterion(aligned_cad_images, ori_images)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * fused_images.size(0)
                total += fused_images.size(0)

            train_loss = running_loss / max(1, total)

            validation_loss = None
            if validation_loader is not None:
                validation_loss = self._evaluate(
                    model, validation_loader, criterion, device, pin_memory
                )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dataset_path": dataset_path,
                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "split_seed": split_seed,
                "split_counts": {
                    "train": len(split_samples["train"]),
                    "validation": len(split_samples["validation"]),
                    "test": len(split_samples["test"]),
                },
                "objective": "self_supervised_cad_to_ori_alignment",
                "loss": "SmoothL1Loss",
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "image_height": ORI_HEIGHT,
                "image_width": ORI_WIDTH,
            }
            torch.save(checkpoint, latest_model_path)

            improved = False
            if validation_loss is not None:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save(checkpoint, best_model_path)
                    improved = True
            elif epoch == epochs:
                torch.save(checkpoint, best_model_path)
                improved = True

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )

            message = f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f}"
            if validation_loss is not None:
                message += f" | val loss {validation_loss:.4f}"
            if improved:
                message += " | checkpoint updated"
            self.signals.progress.emit(message)

        test_loss = None
        if test_loader is not None:
            test_loss = self._evaluate(model, test_loader, criterion, device, pin_memory)
            self.signals.progress.emit(f"Test loss: {test_loss:.4f}")
            self._write_test_results(
                model,
                test_loader,
                test_dataset,
                criterion,
                device,
                pin_memory,
                os.path.join(output_dir, "test_results"),
            )

        self.signals.finished.emit(
            {
                "latest_model_path": latest_model_path,
                "best_model_path": best_model_path,
                "metadata_path": metadata_path,
                "history": history,
                "test_loss": test_loss,
            }
        )

    def _evaluate(self, model, data_loader, criterion, device, pin_memory):
        model.eval()
        total_loss = 0.0
        total = 0

        with torch.no_grad():
            for fused_images, cad_images, ori_images in data_loader:
                fused_images = fused_images.to(device, non_blocking=pin_memory)
                cad_images = cad_images.to(device, non_blocking=pin_memory)
                ori_images = ori_images.to(device, non_blocking=pin_memory)
                aligned_cad_images = model(fused_images, cad_images)
                loss = criterion(aligned_cad_images, ori_images)
                total_loss += loss.item() * fused_images.size(0)
                total += fused_images.size(0)

        return total_loss / max(1, total)

    def _write_test_results(self, model, data_loader, dataset, criterion, device, pin_memory, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        sample_offset = 0
        result_batch = []
        self.signals.progress.emit("Writing STN-transformed test outputs...")

        with torch.no_grad():
            for fused_images, cad_images, ori_images in data_loader:
                batch_size = fused_images.size(0)
                fused_images = fused_images.to(device, non_blocking=pin_memory)
                cad_images = cad_images.to(device, non_blocking=pin_memory)
                ori_images = ori_images.to(device, non_blocking=pin_memory)
                aligned_cad_images = model(fused_images, cad_images).clamp(0.0, 1.0)
                per_item_losses = F.smooth_l1_loss(
                    aligned_cad_images,
                    ori_images,
                    reduction="none",
                ).mean(dim=(1, 2, 3))

                for item_index in range(batch_size):
                    sample_index = sample_offset + item_index
                    cad_path, ori_path = dataset.samples[sample_index]
                    output_path = os.path.join(output_dir, f"test_{sample_index + 1:06d}_aligned_cad.png")
                    save_image(aligned_cad_images[item_index].cpu(), output_path)
                    result_batch.append(
                        {
                            "index": sample_index + 1,
                            "cad_path": cad_path,
                            "ori_path": ori_path,
                            "aligned_path": output_path,
                            "loss": float(per_item_losses[item_index].detach().cpu().item()),
                        }
                    )

                    if len(result_batch) >= 64:
                        self.signals.test_result_batch.emit(result_batch)
                        result_batch = []

                sample_offset += batch_size

        if result_batch:
            self.signals.test_result_batch.emit(result_batch)
        self.signals.progress.emit(f"STN-transformed test outputs saved to: {output_dir}")


class ExportOnnxTask(QRunnable):
    def __init__(self, checkpoint_path, output_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.signals = ExportSignals()

    def run(self):
        if torch is None:
            self.signals.failed.emit(
                f"PyTorch is not available. Install dependencies first.\n{TORCH_IMPORT_ERROR or ''}".strip()
            )
            return

        try:
            self._export()
        except Exception:
            self.signals.failed.emit(traceback.format_exc())

    def _export(self):
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.signals.started.emit("Exporting ONNX model...")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        image_height = checkpoint.get("image_height", ORI_HEIGHT)
        image_width = checkpoint.get("image_width", ORI_WIDTH)

        model = STNAligner()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        dummy_input = torch.randn(1, 6, image_height, image_width)

        torch.onnx.export(
            model,
            dummy_input,
            self.output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["aligned_cad"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "aligned_cad": {0: "batch_size"},
            },
        )

        onnx_results_dir = os.path.join(os.path.dirname(self.output_path), "onnx_results")
        onnx_result_count = self._write_onnx_test_results(checkpoint, onnx_results_dir)

        self.signals.finished.emit(
            {
                "checkpoint_path": self.checkpoint_path,
                "onnx_path": self.output_path,
                "onnx_results_dir": onnx_results_dir,
                "onnx_result_count": onnx_result_count,
            }
        )

    def _write_onnx_test_results(self, checkpoint, output_dir):
        if ort is None:
            self.signals.progress.emit(
                f"ONNX Runtime is not available, so image verification was skipped. {ONNXRUNTIME_IMPORT_ERROR or ''}".strip()
            )
            return 0

        dataset_path = checkpoint.get("dataset_path")
        if not dataset_path or not os.path.isdir(dataset_path):
            self.signals.progress.emit("ONNX image verification skipped: dataset path is missing from checkpoint.")
            return 0

        all_samples = collect_paired_samples(dataset_path)
        split_samples = split_paired_samples(
            all_samples,
            checkpoint.get("train_ratio", 80.0),
            checkpoint.get("validation_ratio", 10.0),
            checkpoint.get("test_ratio", 10.0),
            seed=checkpoint.get("split_seed", 42),
        )
        test_samples = split_samples["test"]
        if not test_samples:
            self.signals.progress.emit("ONNX image verification skipped: test split is empty.")
            return 0

        os.makedirs(output_dir, exist_ok=True)
        session = ort.InferenceSession(self.output_path, providers=["CPUExecutionProvider"])
        dataset = RecursiveImageDataset(
            dataset_path,
            samples=test_samples,
            rotation=None,
            cache_mode="processed",
        )
        result_batch = []
        self.signals.progress.emit("Running ONNX model on test split images...")

        for sample_index in range(len(dataset)):
            fused_tensor, _cad_tensor, ori_tensor = dataset[sample_index]
            onnx_input = fused_tensor.unsqueeze(0).numpy().astype("float32")
            onnx_output = session.run(["aligned_cad"], {"input": onnx_input})[0][0]
            output_tensor = torch.from_numpy(onnx_output).clamp(0.0, 1.0)
            output_path = os.path.join(output_dir, f"onnx_{sample_index + 1:06d}_aligned_cad.png")
            save_image(output_tensor, output_path)

            ori_path = test_samples[sample_index][1]
            cad_path = test_samples[sample_index][0]
            loss = F.smooth_l1_loss(output_tensor, ori_tensor).item()
            result_batch.append(
                {
                    "index": sample_index + 1,
                    "cad_path": cad_path,
                    "ori_path": ori_path,
                    "aligned_path": output_path,
                    "loss": float(loss),
                }
            )

            if len(result_batch) >= 64:
                self.signals.onnx_result_batch.emit(result_batch)
                result_batch = []

        if result_batch:
            self.signals.onnx_result_batch.emit(result_batch)

        self.signals.progress.emit(f"ONNX-transformed test outputs saved to: {output_dir}")
        return len(test_samples)
