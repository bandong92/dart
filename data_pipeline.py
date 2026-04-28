import hashlib
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
CACHE_DIR_NAME = "_dart_cache"


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
    return build_processed_pair_tensors(cad_image, ori_image, rotation_transform)


def build_processed_pair_tensors(cad_image, ori_image, rotation_transform=None):
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
    def __init__(
        self,
        root_dir,
        samples=None,
        image_size=DEFAULT_IMAGE_SIZE,
        rotation=None,
        cache_mode="none",
    ):
        self.root_dir = root_dir
        self.cad_root = os.path.join(root_dir, "cad")
        self.ori_root = os.path.join(root_dir, "ori")
        self.samples = list(samples) if samples is not None else collect_paired_samples(root_dir)

        self.rotation = rotation
        self.cache_mode = cache_mode
        self.cache_dir = os.path.join(root_dir, CACHE_DIR_NAME, cache_mode)
        if self.cache_mode != "none":
            os.makedirs(self.cache_dir, exist_ok=True)

        if not self.samples:
            raise ValueError(f"No images were found in: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cad_path, ori_path = self.samples[index]
        if self.cache_mode == "processed":
            cad_tensor, ori_tensor, fused_tensor = self._load_or_build_processed_cache(
                index, cad_path, ori_path
            )
            return fused_tensor, cad_tensor, ori_tensor

        if self.cache_mode == "source":
            cad_tensor, ori_tensor = self._load_or_build_source_cache(index, cad_path, ori_path)
        else:
            cad_tensor, ori_tensor = build_processed_pair(cad_path, ori_path, self.rotation)
        fused_tensor = self._concat_channels(cad_tensor, ori_tensor)

        return fused_tensor, cad_tensor, ori_tensor

    def _concat_channels(self, cad_tensor, ori_tensor):
        return torch.cat([cad_tensor, ori_tensor], dim=0)

    def _load_or_build_processed_cache(self, index, cad_path, ori_path):
        cache_path = self._cache_path(index, cad_path, ori_path, suffix="processed")
        if os.path.isfile(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            return cached["cad"], cached["ori"], cached["fused"]

        cad_tensor, ori_tensor = build_processed_pair(cad_path, ori_path, self.rotation)
        fused_tensor = self._concat_channels(cad_tensor, ori_tensor)
        torch.save({"cad": cad_tensor, "ori": ori_tensor, "fused": fused_tensor}, cache_path)
        return cad_tensor, ori_tensor, fused_tensor

    def _load_or_build_source_cache(self, index, cad_path, ori_path):
        cache_path = self._cache_path(index, cad_path, ori_path, suffix="source")
        if os.path.isfile(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            cad_image = cached["cad"]
            ori_image = cached["ori"]
        else:
            cad_image = read_image(cad_path, mode=ImageReadMode.RGB)
            ori_image = read_image(ori_path, mode=ImageReadMode.RGB)
            torch.save({"cad": cad_image, "ori": ori_image}, cache_path)

        return build_processed_pair_tensors(cad_image, ori_image, self.rotation)

    def _cache_path(self, index, cad_path, ori_path, suffix):
        cache_key = self._cache_key(cad_path, ori_path, suffix)
        return os.path.join(self.cache_dir, f"{index:08d}_{cache_key}.pt")

    def _cache_key(self, cad_path, ori_path, suffix):
        cad_stat = os.stat(cad_path)
        ori_stat = os.stat(ori_path)
        payload = "|".join(
            [
                os.path.relpath(cad_path, self.root_dir),
                str(cad_stat.st_mtime_ns),
                str(cad_stat.st_size),
                os.path.relpath(ori_path, self.root_dir),
                str(ori_stat.st_mtime_ns),
                str(ori_stat.st_size),
                str(self.rotation),
                suffix,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_dataloader(dataset, batch_size, shuffle, pin_memory, num_workers=None):
    cpu_count = os.cpu_count() or 1
    if num_workers is None:
        num_workers = min(16, max(0, cpu_count - 1))
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
