import os
import random

try:
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except Exception:
    TF = None
    Image = None
    DataLoader = None
    Dataset = object
    transforms = None


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_IMAGE_SIZE = 128
ORI_HEIGHT = 480
ORI_WIDTH = 640


def discover_class_names(ori_root):
    classes = []
    for entry in sorted(os.scandir(ori_root), key=lambda item: item.name.lower()):
        if entry.is_dir():
            classes.append(entry.name)
    return classes


def sorted_image_files(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            extension = os.path.splitext(file_name)[1].lower()
            if extension in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, file_name))

    image_paths.sort(key=lambda path: os.path.relpath(path, root_dir).lower())
    return image_paths


def collect_paired_samples(root_dir):
    cad_root = os.path.join(root_dir, "cad")
    ori_root = os.path.join(root_dir, "ori")

    if not os.path.isdir(cad_root) or not os.path.isdir(ori_root):
        raise ValueError(f"Expected 'cad' and 'ori' folders under: {root_dir}")

    classes = discover_class_names(ori_root)
    samples = []

    if classes:
        for class_idx, class_name in enumerate(classes):
            ori_class_dir = os.path.join(ori_root, class_name)
            cad_class_dir = os.path.join(cad_root, class_name)
            if not os.path.isdir(ori_class_dir) or not os.path.isdir(cad_class_dir):
                continue
            _append_paired_samples(samples, ori_class_dir, cad_class_dir, class_idx, class_name)
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
    else:
        _append_paired_samples(samples, ori_root, cad_root, 0, "default")
        class_to_idx = {"default": 0}

    return samples, class_to_idx


def _append_paired_samples(samples, ori_dir, cad_dir, class_idx, class_name):
    ori_files = sorted_image_files(ori_dir)
    cad_files = sorted_image_files(cad_dir)

    if len(ori_files) != len(cad_files):
        raise ValueError(
            f"CAD/ORI file count mismatch in group '{class_name}': "
            f"{len(cad_files)} cad files vs {len(ori_files)} ori files."
        )

    for cad_path, ori_path in zip(cad_files, ori_files):
        samples.append((cad_path, ori_path, class_idx))


def build_processed_pair(cad_path, ori_path, rotation_transform=None):
    with Image.open(cad_path) as cad_image, Image.open(ori_path) as ori_image:
        cad_image = cad_image.convert("RGB")
        ori_image = ori_image.convert("RGB")

        ori_width, ori_height = ori_image.size
        if (ori_height, ori_width) != (ORI_HEIGHT, ORI_WIDTH):
            raise ValueError(
                f"ORI image must be {ORI_HEIGHT}x{ORI_WIDTH}, got {ori_height}x{ori_width}: {ori_path}"
            )

        if rotation_transform is not None:
            cad_image = rotate_cad_image(cad_image, rotation_transform)
        if cad_image.height < ori_height or cad_image.width < ori_width:
            raise ValueError(
                f"CAD image must be larger than ORI image for center crop: {cad_path}"
            )

        crop_size = (ori_height, ori_width)
        cad_image = TF.center_crop(cad_image, crop_size)
        return cad_image.copy(), ori_image.copy()


def rotate_cad_image(cad_image, rotation_transform):
    angle = _resolve_rotation_angle(rotation_transform)
    interpolation = getattr(transforms, "InterpolationMode", None)
    bilinear = interpolation.BILINEAR if interpolation is not None else Image.BILINEAR
    return TF.rotate(
        cad_image,
        angle=angle,
        interpolation=bilinear,
        expand=True,
        fill=(0, 0, 0),
    )


def _resolve_rotation_angle(rotation_transform):
    if isinstance(rotation_transform, (int, float)):
        return float(rotation_transform)

    if isinstance(rotation_transform, (tuple, list)) and len(rotation_transform) == 2:
        return random.uniform(float(rotation_transform[0]), float(rotation_transform[1]))

    degrees = getattr(rotation_transform, "degrees", None)
    if isinstance(degrees, (tuple, list)) and len(degrees) == 2:
        return random.uniform(float(degrees[0]), float(degrees[1]))
    if isinstance(degrees, (int, float)):
        value = abs(float(degrees))
        return random.uniform(-value, value)

    raise ValueError("Unsupported rotation specification for CAD image rotation.")


class RecursiveImageDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None, image_size=DEFAULT_IMAGE_SIZE):
        self.root_dir = root_dir
        self.cad_root = os.path.join(root_dir, "cad")
        self.ori_root = os.path.join(root_dir, "ori")
        self.samples = []
        self.has_class_subdirs = False

        if class_to_idx is None:
            self.samples, self.class_to_idx = collect_paired_samples(root_dir)
        else:
            self.class_to_idx = dict(class_to_idx)
            self.samples, _ = collect_paired_samples(root_dir)

        self.to_tensor = transforms.ToTensor()
        self.rotation = (-10.0, 10.0)

        if not self.samples:
            raise ValueError(f"No images were found in: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cad_path, ori_path, target = self.samples[index]
        cad_image, ori_image = build_processed_pair(cad_path, ori_path, self.rotation)
        cad_tensor = self.to_tensor(cad_image)
        ori_tensor = self.to_tensor(ori_image)
        fused_tensor = self._concat_channels(cad_tensor, ori_tensor)

        return fused_tensor, cad_tensor, ori_tensor, target

    def _concat_channels(self, cad_tensor, ori_tensor):
        import torch

        return torch.cat([cad_tensor, ori_tensor], dim=0)


def build_dataloader(dataset, batch_size, shuffle, pin_memory):
    num_workers = min(4, max(0, (os.cpu_count() or 1) - 1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
