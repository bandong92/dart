import os
from collections import deque

from PyQt5.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFrame,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from data_pipeline import collect_paired_samples
from training import ExportOnnxTask, TORCH_IMPORT_ERROR, TrainingTask, torch


PREVIEW_SAMPLE_LIMIT = 48
VALIDATION_BROWSER_BATCH_SIZE = 256
SUPPORTED_DATASETS = ("train", "validation", "test")


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background: #111315;
    color: #ece7df;
    font-family: "Yu Gothic UI";
    font-size: 10.5pt;
}
QLabel#heroTitle {
    font-size: 14pt;
    font-weight: 700;
    color: #f4efe7;
    letter-spacing: 0.3px;
}
QFrame#heroCard,
QGroupBox,
QFrame#previewCard,
QTabWidget::pane {
    background: #181b1f;
    border: 1px solid #272b31;
    border-radius: 20px;
}
QTabWidget::pane {
    margin-top: 6px;
}
QTabBar::tab {
    background: #15181b;
    color: #b8aa95;
    border: 1px solid #272b31;
    padding: 8px 16px;
    min-width: 110px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    margin-right: 6px;
}
QTabBar::tab:selected {
    background: #1f2328;
    color: #f3eee7;
    border-color: #4a433b;
}
QTabBar::tab:hover {
    background: #1b1f23;
}
QGroupBox {
    margin-top: 14px;
    padding: 18px 16px 16px 16px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #b8aa95;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QListWidget {
    background: #121518;
    border: 1px solid #2b3037;
    border-radius: 12px;
    padding: 9px 12px;
    color: #f3eee7;
    selection-background-color: #5c4c3b;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus {
    border: 1px solid #b48a5a;
}
QPushButton {
    background: #1a1d21;
    border: 1px solid #30353d;
    border-radius: 12px;
    padding: 10px 16px;
    color: #f2ede6;
    font-weight: 600;
}
QPushButton:hover {
    background: #242930;
    border-color: #4a433b;
}
QPushButton:pressed {
    background: #101215;
}
QPushButton:disabled {
    background: #14171a;
    color: #6e685f;
    border-color: #23272c;
}
QPushButton#primaryButton {
    background: #8d6745;
    border-color: #af845b;
}
QPushButton#primaryButton:hover {
    background: #9b7350;
}
QPushButton#ghostButton {
    background: #171a1d;
}
QListWidget {
    outline: none;
}
QListWidget::item {
    padding: 8px 10px;
    border-radius: 8px;
    margin: 2px 0;
}
QListWidget::item:selected {
    background: #2a2621;
    color: #fbf7f1;
}
QListWidget::item:hover {
    background: #202429;
}
QSplitter::handle {
    background: #111315;
    width: 12px;
}
QScrollBar:vertical {
    background: #15181b;
    width: 12px;
    margin: 4px;
}
QScrollBar::handle:vertical {
    background: #3a3f46;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


class DatasetScanSignals(QObject):
    progress = pyqtSignal(str, int, int)
    sample = pyqtSignal(str, str, str)
    finished = pyqtSignal(str, int)
    failed = pyqtSignal(str, str)


class DatasetScanTask(QRunnable):
    def __init__(self, dataset_name, folder_path):
        super().__init__()
        self.dataset_name = dataset_name
        self.folder_path = folder_path
        self.signals = DatasetScanSignals()

    def run(self):
        try:
            pairs, _ = collect_paired_samples(self.folder_path)
            total_count = len(pairs)
            sampled_pairs = deque(maxlen=PREVIEW_SAMPLE_LIMIT)

            for index, (cad_path, ori_path, _) in enumerate(pairs, start=1):
                if len(sampled_pairs) < PREVIEW_SAMPLE_LIMIT:
                    sampled_pairs.append((cad_path, ori_path))
                    self.signals.sample.emit(self.dataset_name, cad_path, ori_path)
                elif index % max(1, total_count // PREVIEW_SAMPLE_LIMIT) == 0:
                    sampled_pairs.append((cad_path, ori_path))

                if index % 200 == 0:
                    self.signals.progress.emit(
                        self.dataset_name, index, len(sampled_pairs)
                    )

            if total_count > PREVIEW_SAMPLE_LIMIT:
                for cad_path, ori_path in list(sampled_pairs):
                    self.signals.sample.emit(self.dataset_name, cad_path, ori_path)

            self.signals.finished.emit(self.dataset_name, total_count)
        except Exception as exc:
            self.signals.failed.emit(self.dataset_name, str(exc))


class ValidationBrowserSignals(QObject):
    batch = pyqtSignal(object)
    finished = pyqtSignal(int)
    failed = pyqtSignal(str)


class ValidationBrowserTask(QRunnable):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.signals = ValidationBrowserSignals()

    def run(self):
        try:
            pairs, _ = collect_paired_samples(self.folder_path)
            batch = []
            for index, (cad_path, ori_path, _label) in enumerate(pairs, start=1):
                batch.append(
                    {
                        "index": index,
                        "cad_path": cad_path,
                        "ori_path": ori_path,
                    }
                )
                if len(batch) >= VALIDATION_BROWSER_BATCH_SIZE:
                    self.signals.batch.emit(batch)
                    batch = []

            if batch:
                self.signals.batch.emit(batch)
            self.signals.finished.emit(len(pairs))
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class DatasetPanel(QGroupBox):
    path_requested = pyqtSignal(str)

    def __init__(self, dataset_name):
        super().__init__(dataset_name.capitalize())
        self.dataset_name = dataset_name
        self.sample_paths = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 12, 8, 8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(0)

        path_row = QHBoxLayout()
        path_row.setSpacing(10)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(f"{self.dataset_name} data folder")
        self.path_edit.setMinimumWidth(220)
        self.path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        browse_button = QPushButton("Browse")
        browse_button.setObjectName("ghostButton")
        browse_button.setMinimumWidth(96)
        browse_button.setMaximumWidth(120)
        browse_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        browse_button.clicked.connect(lambda: self.path_requested.emit(self.dataset_name))
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_button)
        path_row.setStretch(0, 1)
        path_row.setStretch(1, 0)
        layout.addLayout(path_row)

        self.status_label = QLabel("No folder selected.")
        self.status_label.setStyleSheet(
            "color: #aea397; background: #121518; border: 1px solid #262b31; border-radius: 10px; padding: 8px 10px;"
        )
        layout.addWidget(self.status_label)

        self.sample_list = QListWidget()
        self.sample_list.setMinimumHeight(90)
        self.sample_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.sample_list)

        self.setTitle("")

    def set_path(self, folder_path):
        self.path_edit.setText(folder_path)

    def set_status(self, message):
        self.status_label.setText(message)

    def clear_samples(self):
        self.sample_paths = []
        self.sample_list.clear()

    def add_sample(self, cad_path, ori_path):
        pair_key = (cad_path, ori_path)
        if pair_key in self.sample_paths:
            return

        self.sample_paths.append(pair_key)
        item = QListWidgetItem(os.path.basename(ori_path))
        item.setToolTip(f"CAD: {cad_path}\nORI: {ori_path}")
        item.setData(Qt.UserRole, pair_key)
        self.sample_list.addItem(item)

        if self.sample_list.count() == 1:
            self.sample_list.setCurrentRow(0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool.globalInstance()
        self.dataset_panels = {}
        self.validation_browser_pixmap_cache = {}
        self.is_training = False
        self.is_exporting = False
        self.onnx_export_ready = False
        self.latest_checkpoint_path = None
        self.best_checkpoint_path = None
        self.validation_browser_item_count = 0
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("DART")
        self.resize(1360, 820)
        self.setMinimumSize(1024, 640)
        self.setStyleSheet(DARK_STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)

        outer_layout = QVBoxLayout(root)
        outer_layout.setContentsMargins(20, 20, 20, 20)
        outer_layout.setSpacing(16)

        hero_card = QFrame()
        hero_card.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(20, 12, 20, 12)
        hero_layout.setSpacing(0)
        hero_title = QLabel("DART")
        hero_title.setObjectName("heroTitle")
        hero_layout.addWidget(hero_title)
        outer_layout.addWidget(hero_card)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer_layout.addWidget(splitter, stretch=1)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        config_group = QGroupBox("Training Configuration")
        config_layout = QFormLayout(config_group)
        config_layout.setSpacing(10)
        config_layout.setLabelAlignment(Qt.AlignLeft)

        self.epoch_input = QSpinBox()
        self.epoch_input.setRange(1, 100000)
        self.epoch_input.setValue(50)
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 100000)
        self.batch_input.setValue(32)
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(6)
        self.lr_input.setRange(0.000001, 10.0)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(0.001)

        config_layout.addRow("Epoch", self.epoch_input)
        config_layout.addRow("Batch Size", self.batch_input)
        config_layout.addRow("Learning Rate", self.lr_input)
        config_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        left_layout.addWidget(config_group, stretch=0)

        dataset_tabs = QTabWidget()
        dataset_tabs.setDocumentMode(True)
        dataset_tabs.setTabPosition(QTabWidget.North)
        dataset_tabs.setMovable(False)
        dataset_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        for dataset_name in SUPPORTED_DATASETS:
            panel = DatasetPanel(dataset_name)
            panel.path_requested.connect(self.select_folder)
            self.dataset_panels[dataset_name] = panel
            dataset_tabs.addTab(panel, dataset_name.capitalize())

        dataset_group = QGroupBox("Dataset Sources")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setContentsMargins(10, 18, 10, 10)
        dataset_layout.addWidget(dataset_tabs)
        dataset_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(dataset_group, stretch=1)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.train_button = QPushButton("Train")
        self.train_button.setObjectName("primaryButton")
        self.train_button.clicked.connect(self.start_training)
        self.onnx_button = QPushButton("Export ONNX")
        self.onnx_button.setObjectName("ghostButton")
        self.onnx_button.setEnabled(False)
        self.onnx_button.clicked.connect(self.export_onnx)
        button_row.addWidget(self.train_button)
        button_row.addWidget(self.onnx_button)
        left_layout.addLayout(button_row)

        self.summary_label = QLabel(
            "Select a valid train dataset folder to enable training."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #9f9589; padding: 4px 2px;")
        left_layout.addWidget(self.summary_label, stretch=0)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(130)
        self.log_view.setStyleSheet(
            "background: #121518; border: 1px solid #262b31; border-radius: 12px; color: #ded6cb; padding: 8px;"
        )
        left_layout.addWidget(self.log_view, stretch=0)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        validation_card = QFrame()
        validation_card.setObjectName("previewCard")
        validation_layout = QVBoxLayout(validation_card)
        validation_layout.setContentsMargins(18, 18, 18, 18)
        validation_layout.setSpacing(12)

        validation_title = QLabel("Validation Browser")
        validation_title.setStyleSheet("font-size: 15pt; font-weight: 700; color: #f4efe7;")
        validation_layout.addWidget(validation_title)

        validation_hint = QLabel(
            "All validation CAD and ORI pairs are listed in scan order. "
            "Only the selected pair is loaded as an image so large datasets stay responsive."
        )
        validation_hint.setWordWrap(True)
        validation_hint.setStyleSheet("color: #a79d90;")
        validation_layout.addWidget(validation_hint)

        self.validation_status_label = QLabel(
            "Select a validation dataset folder to populate this browser."
        )
        self.validation_status_label.setWordWrap(True)
        self.validation_status_label.setStyleSheet(
            "color: #aea397; background: #121518; border: 1px solid #262b31; border-radius: 10px; padding: 8px 10px;"
        )
        validation_layout.addWidget(self.validation_status_label)

        self.validation_browser_list = QListWidget()
        self.validation_browser_list.setMinimumHeight(220)
        self.validation_browser_list.setUniformItemSizes(True)
        self.validation_browser_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.validation_browser_list.currentItemChanged.connect(self.on_validation_browser_selected)
        validation_layout.addWidget(self.validation_browser_list, stretch=1)

        image_row = QHBoxLayout()
        image_row.setSpacing(12)
        self.validation_browser_cad_label = self._create_preview_panel("Validation CAD")
        self.validation_browser_ori_label = self._create_preview_panel("Validation ORI")
        image_row.addWidget(self.validation_browser_cad_label, stretch=1)
        image_row.addWidget(self.validation_browser_ori_label, stretch=1)
        validation_layout.addLayout(image_row, stretch=1)

        self.validation_browser_meta_label = QLabel("Validation pair metadata will appear here.")
        self.validation_browser_meta_label.setWordWrap(True)
        self.validation_browser_meta_label.setStyleSheet(
            "color: #a99d91; background: #121518; border: 1px solid #262b31; border-radius: 12px; padding: 10px 12px;"
        )
        validation_layout.addWidget(self.validation_browser_meta_label)

        right_layout.addWidget(validation_card, stretch=1)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        left_widget.setMinimumWidth(460)
        right_widget.setMinimumWidth(520)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([620, 740])
        self.refresh_action_state()

    def select_folder(self, dataset_name):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            f"Select {dataset_name} dataset folder",
            self.dataset_panels[dataset_name].path_edit.text() or os.getcwd(),
        )
        if not folder_path:
            return

        panel = self.dataset_panels[dataset_name]
        panel.set_path(folder_path)
        panel.clear_samples()
        panel.set_status("Scanning in background...")

        task = DatasetScanTask(dataset_name, folder_path)
        task.signals.progress.connect(self.on_scan_progress)
        task.signals.sample.connect(self.on_scan_sample)
        task.signals.finished.connect(self.on_scan_finished)
        task.signals.failed.connect(self.on_scan_failed)
        self.thread_pool.start(task)

        if dataset_name == "validation":
            self.start_validation_browser_scan(folder_path)

        self.refresh_action_state()

    def start_validation_browser_scan(self, folder_path):
        self.clear_validation_browser()
        self.validation_status_label.setText("Scanning validation pairs in background...")
        task = ValidationBrowserTask(folder_path)
        task.signals.batch.connect(self.on_validation_browser_batch)
        task.signals.finished.connect(self.on_validation_browser_finished)
        task.signals.failed.connect(self.on_validation_browser_failed)
        self.thread_pool.start(task)

    def on_scan_progress(self, dataset_name, count, sample_count):
        self.dataset_panels[dataset_name].set_status(
            f"Scanning... {count:,} images found, {sample_count} samples loaded."
        )

    def on_scan_sample(self, dataset_name, cad_path, ori_path):
        self.dataset_panels[dataset_name].add_sample(cad_path, ori_path)

    def on_scan_finished(self, dataset_name, count):
        self.dataset_panels[dataset_name].set_status(
            f"Ready. {count:,} images detected. Showing sampled previews only for smooth performance."
        )
        self.refresh_action_state()

    def on_scan_failed(self, dataset_name, error_message):
        self.dataset_panels[dataset_name].set_status("Scan failed.")
        QMessageBox.critical(self, f"{dataset_name.capitalize()} scan failed", error_message)
        self.refresh_action_state()

    def start_training(self):
        if self.is_training:
            return

        if torch is None:
            QMessageBox.critical(
                self,
                "PyTorch unavailable",
                f"PyTorch import failed.\n\n{TORCH_IMPORT_ERROR or 'Unknown error'}",
            )
            return

        train_path = self.dataset_panels["train"].path_edit.text().strip()
        validation_path = self.dataset_panels["validation"].path_edit.text().strip()
        test_path = self.dataset_panels["test"].path_edit.text().strip()

        if not train_path or not os.path.isdir(train_path):
            QMessageBox.warning(self, "Missing train path", "Select a valid train dataset folder first.")
            return

        config = {
            "train_path": train_path,
            "validation_path": validation_path,
            "test_path": test_path,
            "epochs": self.epoch_input.value(),
            "batch_size": self.batch_input.value(),
            "learning_rate": self.lr_input.value(),
        }

        self.is_training = True
        self.onnx_export_ready = False
        self.latest_checkpoint_path = None
        self.best_checkpoint_path = None
        self.log_view.clear()
        self.append_log("Starting training...")
        self.summary_label.setText("Training in progress. Checkpoints will be saved automatically to outputs.")
        self.refresh_action_state()

        task = TrainingTask(config)
        task.signals.started.connect(self.append_log)
        task.signals.progress.connect(self.append_log)
        task.signals.validation_preview.connect(self.on_validation_preview)
        task.signals.finished.connect(self.on_training_finished)
        task.signals.failed.connect(self.on_training_failed)
        self.thread_pool.start(task)

    def on_training_finished(self, result):
        self.is_training = False
        self.onnx_export_ready = True
        self.latest_checkpoint_path = result["latest_model_path"]
        self.best_checkpoint_path = result["best_model_path"]
        self.append_log(f"Latest checkpoint saved: {result['latest_model_path']}")
        self.append_log(f"Best checkpoint saved: {result['best_model_path']}")
        self.append_log(f"Training config saved: {result['metadata_path']}")
        self.summary_label.setText("Training finished. Checkpoints were saved automatically to outputs.")
        self.refresh_action_state()

    def on_training_failed(self, error_message):
        self.is_training = False
        self.onnx_export_ready = False
        self.append_log("Training failed.")
        self.append_log(error_message.strip())
        self.summary_label.setText("Training failed. Review the log output and dataset structure.")
        QMessageBox.critical(self, "Training failed", error_message)
        self.refresh_action_state()

    def export_onnx(self):
        if self.is_training or self.is_exporting:
            return

        checkpoint_path = self.best_checkpoint_path or self.latest_checkpoint_path
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            QMessageBox.warning(
                self,
                "No checkpoint found",
                "Train the model first so there is a checkpoint to export.",
            )
            return

        output_dir = os.path.join(os.getcwd(), "outputs")
        output_path = os.path.join(output_dir, "dart_stn.onnx")

        self.is_exporting = True
        self.append_log("Starting ONNX export...")
        self.summary_label.setText("Exporting the current best checkpoint to ONNX.")
        self.refresh_action_state()

        task = ExportOnnxTask(checkpoint_path, output_path)
        task.signals.started.connect(self.append_log)
        task.signals.finished.connect(self.on_export_finished)
        task.signals.failed.connect(self.on_export_failed)
        self.thread_pool.start(task)

    def on_export_finished(self, result):
        self.is_exporting = False
        self.append_log(f"ONNX export complete: {result['onnx_path']}")
        self.summary_label.setText("ONNX export finished successfully.")
        QMessageBox.information(self, "Export complete", f"ONNX saved to:\n{result['onnx_path']}")
        self.refresh_action_state()

    def on_export_failed(self, error_message):
        self.is_exporting = False
        self.append_log("ONNX export failed.")
        self.append_log(error_message.strip())
        self.summary_label.setText("ONNX export failed. Review the log output.")
        QMessageBox.critical(self, "ONNX export failed", error_message)
        self.refresh_action_state()

    def append_log(self, message):
        self.log_view.append(message)

    def on_validation_preview(self, payload):
        record = {
            "index": payload.get("index", self.validation_browser_item_count + 1),
            "cad_path": payload.get("cad_path", ""),
            "ori_path": payload.get("ori_path", ""),
            "title": payload.get("title"),
            "tooltip": payload.get("tooltip"),
            "meta": payload.get("meta"),
            "auto_select": payload.get("auto_select", True),
        }
        self._append_validation_browser_entry(record)

    def on_validation_browser_batch(self, batch):
        for entry in batch:
            self._append_validation_browser_entry(entry, auto_select=False)

        self.validation_status_label.setText(
            f"Loading validation pairs... {self.validation_browser_list.count():,} items ready."
        )
        if self.validation_browser_list.count() == len(batch):
            self.validation_browser_list.setCurrentRow(0)

    def on_validation_browser_finished(self, total_count):
        self.validation_status_label.setText(
            f"Validation browser ready. {total_count:,} pairs available in scroll view."
        )

    def on_validation_browser_failed(self, error_message):
        self.validation_status_label.setText("Validation browser scan failed.")
        QMessageBox.critical(self, "Validation browser scan failed", error_message)

    def on_validation_browser_selected(self, current_item, _previous_item):
        if not current_item:
            return

        entry = current_item.data(Qt.UserRole)
        cad_path = entry["cad_path"]
        ori_path = entry["ori_path"]
        self.validation_browser_meta_label.setText(
            f"Index: {entry['index']}\nCAD: {cad_path}\nORI: {ori_path}"
        )

        cad_pixmap = self._get_validation_browser_pixmap(cad_path)
        ori_pixmap = self._get_validation_browser_pixmap(ori_path)
        self._set_preview_pixmap(self.validation_browser_cad_label, cad_pixmap, "Validation CAD")
        self._set_preview_pixmap(self.validation_browser_ori_label, ori_pixmap, "Validation ORI")

    def clear_validation_browser(self):
        self.validation_browser_item_count = 0
        self.validation_browser_list.clear()
        self.validation_browser_pixmap_cache.clear()
        self.validation_status_label.setText("Select a validation dataset folder to populate this browser.")
        self.validation_browser_meta_label.setText("Validation pair metadata will appear here.")
        self.validation_browser_cad_label.setText("Validation CAD")
        self.validation_browser_cad_label.setPixmap(QPixmap())
        self.validation_browser_ori_label.setText("Validation ORI")
        self.validation_browser_ori_label.setPixmap(QPixmap())

    def refresh_action_state(self):
        train_ready = os.path.isdir(self.dataset_panels["train"].path_edit.text().strip())
        busy = self.is_training or self.is_exporting
        self.train_button.setEnabled(train_ready and not busy)
        self.onnx_button.setEnabled(self.onnx_export_ready and not busy)

        if self.is_training:
            status_message = "Training in progress. Check the log for epoch updates."
            train_tooltip = "Training is currently running."
            onnx_tooltip = "ONNX export is available after training finishes."
        elif self.is_exporting:
            status_message = "ONNX export in progress."
            train_tooltip = "Wait for ONNX export to finish."
            onnx_tooltip = "ONNX export is currently running."
        elif not train_ready:
            status_message = (
                "Train is disabled: select a valid train dataset folder with cad and ori subfolders."
            )
            train_tooltip = "Set the Train dataset path to a valid folder first."
            onnx_tooltip = "Run training first to create a checkpoint for ONNX export."
        elif not self.onnx_export_ready:
            status_message = (
                "Train is ready. ONNX export is disabled until a training run creates a checkpoint."
            )
            train_tooltip = "Start training with the current dataset paths and options."
            onnx_tooltip = "Run training first to enable ONNX export."
        else:
            status_message = "Train is ready, and ONNX export is available from the latest trained checkpoint."
            train_tooltip = "Start a new training run."
            onnx_tooltip = "Export the best saved checkpoint to ONNX."

        self.summary_label.setText(status_message)
        self.train_button.setToolTip(train_tooltip)
        self.onnx_button.setToolTip(onnx_tooltip)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        current_browser_item = self.validation_browser_list.currentItem()
        if current_browser_item is not None:
            self.on_validation_browser_selected(current_browser_item, None)

    def _create_preview_panel(self, placeholder_text):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(250, 210)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setStyleSheet(
            "QLabel { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #121417, stop:1 #1c2025); color: #ddd5cb; border: 1px solid #2b3037; border-radius: 18px; }"
        )
        label.setText(placeholder_text)
        return label

    def _load_preview_pixmap(self, image_path):
        if not image_path or not os.path.isfile(image_path):
            return None

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return None
        return pixmap

    def _get_validation_browser_pixmap(self, image_path):
        if image_path in self.validation_browser_pixmap_cache:
            return self.validation_browser_pixmap_cache[image_path]

        pixmap = self._load_preview_pixmap(image_path)
        if pixmap is None:
            return None

        if len(self.validation_browser_pixmap_cache) >= 16:
            oldest_key = next(iter(self.validation_browser_pixmap_cache))
            self.validation_browser_pixmap_cache.pop(oldest_key, None)

        self.validation_browser_pixmap_cache[image_path] = pixmap
        return pixmap

    def _append_validation_browser_entry(self, entry, auto_select=None):
        self.validation_browser_item_count += 1
        item_title = entry.get("title") or f"{entry['index']:06d} | {os.path.basename(entry['ori_path'])}"
        item = QListWidgetItem(item_title)
        item.setToolTip(
            entry.get("tooltip") or f"CAD: {entry['cad_path']}\nORI: {entry['ori_path']}"
        )
        item.setData(Qt.UserRole, entry)
        self.validation_browser_list.addItem(item)

        should_select = entry.get("auto_select", True) if auto_select is None else auto_select
        if self.validation_browser_list.count() == 1 or should_select:
            self.validation_browser_list.setCurrentRow(self.validation_browser_list.count() - 1)

    def _set_preview_pixmap(self, label, pixmap, fallback_text):
        if pixmap is None:
            label.setPixmap(QPixmap())
            label.setText(fallback_text)
            return

        scaled = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setText("")
        label.setPixmap(scaled)
