import os
import random

try:
    import torch
    import torchvision.transforms.functional as TF
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.io import ImageReadMode, read_image
except Exception:
    torch = None
    TF = None
    DataLoader = None
    Dataset = object
    transforms = None
    ImageReadMode = None
    read_image = None


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

    samples = []
    ori_files = sorted_image_files(ori_root)
    cad_files = sorted_image_files(cad_root)

    if len(ori_files) != len(cad_files):
        raise ValueError(
            f"CAD/ORI file count mismatch: {len(cad_files)} cad files vs "
            f"{len(ori_files)} ori files under {root_dir}."
        )

    for cad_path, ori_path in zip(cad_files, ori_files):
        samples.append((cad_path, ori_path))

    return samples


def split_paired_samples(samples, train_ratio, validation_ratio, test_ratio, seed=42):
    total_ratio = train_ratio + validation_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be greater than 0.")

    shuffled_samples = list(samples)
    random.Random(seed).shuffle(shuffled_samples)

    total_count = len(shuffled_samples)
    train_count = int(total_count * (train_ratio / total_ratio))
    validation_count = int(total_count * (validation_ratio / total_ratio))

    train_samples = shuffled_samples[:train_count]
    validation_samples = shuffled_samples[train_count:train_count + validation_count]
    test_samples = shuffled_samples[train_count + validation_count:]

    return {
        "train": train_samples,
        "validation": validation_samples,
        "test": test_samples,
    }


def build_processed_pair(cad_path, ori_path, rotation_transform=None):
    cad_image = read_image(cad_path, mode=ImageReadMode.RGB)
    ori_image = read_image(ori_path, mode=ImageReadMode.RGB)
    cad_image = TF.convert_image_dtype(cad_image, torch.float32)
    ori_image = TF.convert_image_dtype(ori_image, torch.float32)

    ori_height, ori_width = ori_image.shape[1:]
    if (ori_height, ori_width) != (ORI_HEIGHT, ORI_WIDTH):
        raise ValueError(
            f"ORI image must be {ORI_HEIGHT}x{ORI_WIDTH}, got {ori_height}x{ori_width}: {ori_path}"
        )

    if rotation_transform is not None:
        cad_image = rotate_cad_image(cad_image, rotation_transform)
    cad_height, cad_width = cad_image.shape[1:]
    if cad_height < ori_height or cad_width < ori_width:
        raise ValueError(
            f"CAD image must be larger than ORI image for center crop: {cad_path}"
        )

    crop_size = (ori_height, ori_width)
    cad_image = TF.center_crop(cad_image, crop_size)
    return cad_image.contiguous(), ori_image.contiguous()


def rotate_cad_image(cad_image, rotation_transform):
    angle = _resolve_rotation_angle(rotation_transform)
    interpolation = getattr(transforms, "InterpolationMode", None)
    bilinear = interpolation.BILINEAR if interpolation is not None else 2
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
    def __init__(self, root_dir, samples=None, image_size=DEFAULT_IMAGE_SIZE):
        self.root_dir = root_dir
        self.cad_root = os.path.join(root_dir, "cad")
        self.ori_root = os.path.join(root_dir, "ori")
        self.samples = list(samples) if samples is not None else collect_paired_samples(root_dir)

        self.rotation = (-10.0, 10.0)

        if not self.samples:
            raise ValueError(f"No images were found in: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cad_path, ori_path = self.samples[index]
        cad_tensor, ori_tensor = build_processed_pair(cad_path, ori_path, self.rotation)
        fused_tensor = self._concat_channels(cad_tensor, ori_tensor)

        return fused_tensor, cad_tensor, ori_tensor

    def _concat_channels(self, cad_tensor, ori_tensor):
        return torch.cat([cad_tensor, ori_tensor], dim=0)


def build_dataloader(dataset, batch_size, shuffle, pin_memory):
    cpu_count = os.cpu_count() or 1
    num_workers = min(8, max(0, cpu_count - 1))
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)
