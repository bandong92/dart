import json
import os
import traceback

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from data_pipeline import ORI_HEIGHT, ORI_WIDTH, RecursiveImageDataset, build_dataloader

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = str(exc)


class TrainingSignals(QObject):
    started = pyqtSignal(str)
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)


class ExportSignals(QObject):
    started = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)


if torch is not None:
    class STNClassifier(nn.Module):
        def __init__(self, num_classes):
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

            self.features = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def stn(self, x):
            xs = self.localization(x)
            xs = xs.view(-1, 10 * 8 * 8)
            theta = self.fc_loc(xs).view(-1, 2, 3)
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            return F.grid_sample(x, grid, align_corners=False)

        def forward(self, x):
            x = self.stn(x)
            x = self.features(x)
            return self.classifier(x)


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
        train_path = self.config["train_path"]
        validation_path = self.config["validation_path"]
        test_path = self.config["test_path"]
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]

        self.signals.started.emit("Preparing datasets...")

        train_dataset = RecursiveImageDataset(train_path)
        validation_dataset = None
        if validation_path and os.path.isdir(validation_path):
            validation_dataset = RecursiveImageDataset(
                validation_path, class_to_idx=train_dataset.class_to_idx
            )

        test_dataset = None
        if test_path and os.path.isdir(test_path):
            test_dataset = RecursiveImageDataset(
                test_path, class_to_idx=train_dataset.class_to_idx
            )

        pin_memory = torch.cuda.is_available()

        train_loader = build_dataloader(train_dataset, batch_size, True, pin_memory)
        validation_loader = None
        if validation_dataset is not None:
            validation_loader = build_dataloader(validation_dataset, batch_size, False, pin_memory)

        test_loader = None
        if test_dataset is not None:
            test_loader = build_dataloader(test_dataset, batch_size, False, pin_memory)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = STNClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        latest_model_path = os.path.join(output_dir, "dart_stn_latest.pt")
        best_model_path = os.path.join(output_dir, "dart_stn_best.pt")
        metadata_path = os.path.join(output_dir, "dart_training_config.json")

        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(
                {
                    "train_path": train_path,
                    "validation_path": validation_path,
                    "test_path": test_path,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "class_to_idx": train_dataset.class_to_idx,
                },
                metadata_file,
                indent=2,
            )

        best_validation_accuracy = -1.0
        history = []
        self.signals.progress.emit(
            f"Training started on {device.type.upper()} with {len(train_dataset):,} training images."
        )

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for fused_images, cad_images, ori_images, labels in train_loader:
                fused_images = fused_images.to(device, non_blocking=pin_memory)
                cad_images = cad_images.to(device, non_blocking=pin_memory)
                ori_images = ori_images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(fused_images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * fused_images.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / max(1, total)
            train_accuracy = correct / max(1, total)

            validation_loss = None
            validation_accuracy = None
            if validation_loader is not None:
                validation_loss, validation_accuracy = self._evaluate(
                    model, validation_loader, criterion, device, pin_memory
                )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "image_height": ORI_HEIGHT,
                "image_width": ORI_WIDTH,
            }
            torch.save(checkpoint, latest_model_path)

            improved = False
            if validation_accuracy is not None:
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    torch.save(checkpoint, best_model_path)
                    improved = True
            elif epoch == epochs:
                torch.save(checkpoint, best_model_path)
                improved = True

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )

            message = f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f} | train acc {train_accuracy:.2%}"
            if validation_accuracy is not None:
                message += f" | val loss {validation_loss:.4f} | val acc {validation_accuracy:.2%}"
            if improved:
                message += " | checkpoint updated"
            self.signals.progress.emit(message)

        test_accuracy = None
        if test_loader is not None:
            _, test_accuracy = self._evaluate(model, test_loader, criterion, device, pin_memory)
            self.signals.progress.emit(f"Test accuracy: {test_accuracy:.2%}")

        self.signals.finished.emit(
            {
                "latest_model_path": latest_model_path,
                "best_model_path": best_model_path,
                "metadata_path": metadata_path,
                "history": history,
                "test_accuracy": test_accuracy,
            }
        )

    def _evaluate(self, model, data_loader, criterion, device, pin_memory):
        model.eval()
        total_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for fused_images, cad_images, ori_images, labels in data_loader:
                fused_images = fused_images.to(device, non_blocking=pin_memory)
                cad_images = cad_images.to(device, non_blocking=pin_memory)
                ori_images = ori_images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                outputs = model(fused_images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * fused_images.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return total_loss / max(1, total), correct / max(1, total)


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
        class_to_idx = checkpoint.get("class_to_idx", {})
        image_height = checkpoint.get("image_height", ORI_HEIGHT)
        image_width = checkpoint.get("image_width", ORI_WIDTH)

        model = STNClassifier(num_classes=len(class_to_idx))
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
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        self.signals.finished.emit(
            {
                "checkpoint_path": self.checkpoint_path,
                "onnx_path": self.output_path,
            }
        )
