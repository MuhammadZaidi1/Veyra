"""Procedural non-face signals and explicit local, registered-mesh manifests."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Manifest paths must be relative to the approved dataset root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its root, including through a symlink")
    if not path.is_file():
        raise ValueError(f"Local dataset file is missing: {path}")
    return path


def preprocess_rgb(image, size):
    """Letterbox without changing aspect; return normalized tensor + XYZ scale.

    Labels start in centered original-crop-width units. Their XYZ coordinates
    all scale by width / max(width,height) into square-canvas-width units.
    """
    if size not in (64, 128, 192, 256):
        raise ValueError("Unsupported input size")
    if image.getexif().get(274, 1) != 1:
        raise ValueError("Resolve image EXIF orientation and label coordinates before import")
    width, height = image.size
    if min(width, height) <= 0 or max(width, height) > 4096:
        raise ValueError("Approved input crops must have sides between 1 and 4096 pixels")
    fit = size / max(width, height)
    new_size = (max(1, round(width * fit)), max(1, round(height * fit)))
    resized = image.convert("RGB").resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    rgb = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous(), width / max(width, height)


class SyntheticDataset(Dataset):
    """Images encoding three coefficients + a 468-node deformed grid, NOT faces."""

    def __init__(self, samples, size, seed):
        self.samples, self.size, self.seed = samples, size, seed
        y, x = torch.meshgrid(torch.linspace(-.5, .5, 18), torch.linspace(-.4, .4, 26), indexing="ij")
        self.grid = torch.stack((x.flatten(), y.flatten()), dim=-1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        self.image_grid = torch.stack((xx, yy, xx * yy))
        self.signature = hashlib.sha256(f"procedural-grid-v2:{samples}:{size}:{seed}".encode()).hexdigest()

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        if not 0 <= index < self.samples:
            raise IndexError(index)
        # Mix split and sample entropy explicitly; high-bit-only torch seeds can
        # repeat CPU random streams on some builds.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        coefficients = torch.from_numpy(rng.uniform(-1, 1, 3).astype(np.float32))
        x, y = self.grid.unbind(dim=-1)
        a, b, c = coefficients
        vertices = torch.stack((x * (1 + .12 * a), y * (1 + .12 * b),
                                .1 * (x*x + y*y) + .05 * c * torch.sin(x*5) * torch.cos(y*4)), dim=-1)
        vertices -= vertices.mean(dim=0, keepdim=True)
        images = (coefficients[:, None, None] * .65 + self.image_grid * .2).float()
        return {"image": images, "vertices": vertices, "valid": torch.ones(468, dtype=torch.bool)}


class ManifestDataset(Dataset):
    """No collection or download. All records reference explicitly local files."""

    def __init__(self, root, manifest, size):
        self.root, self.size = Path(root).resolve(), size
        self.manifest = local_path(self.root, manifest)
        self.records, self.groups, self.image_paths, self.image_hashes = [], set(), set(), set()
        digest = hashlib.sha256(self.manifest.read_bytes())
        for line_number, line in enumerate(self.manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if set(record) != {"image", "geometry", "group"}:
                    raise ValueError("Each record requires exactly image, geometry and opaque split group")
                if not isinstance(record["group"], str) or not record["group"].strip():
                    raise ValueError("group must be a nonempty de-identified split key")
                image = local_path(self.root, record["image"])
                geometry = local_path(self.root, record["geometry"])
                if image in self.image_paths:
                    raise ValueError("Duplicate image record")
                image_hash = sha256_file(image)
                if image_hash in self.image_hashes:
                    raise ValueError("Duplicate image content")
                self.records.append((image, geometry))
                self.groups.add(record["group"])
                self.image_paths.add(image)
                self.image_hashes.add(image_hash)
                # Content fingerprints make changed files detectable at resume.
                digest.update(image_hash.encode())
                digest.update(sha256_file(geometry).encode())
            except (TypeError, KeyError, ValueError) as error:
                raise ValueError(f"Invalid manifest record at line {line_number}: {error}") from error
        if not self.records:
            raise ValueError("Dataset manifest is empty")
        self.signature = digest.hexdigest()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, geometry_path = self.records[index]
        with Image.open(image_path) as source:
            image, scale = preprocess_rgb(source, self.size)
        with np.load(geometry_path, allow_pickle=False) as source:
            if set(source.files) != {"vertices", "valid"}:
                raise ValueError("Geometry NPZ requires exactly vertices and valid arrays")
            vertices = source["vertices"]
            valid = source["valid"]
        if vertices.shape != (468, 3) or vertices.dtype != np.float32:
            raise ValueError("vertices must be a float32 [468,3] array in the documented coordinate frame")
        if valid.shape != (468,) or valid.dtype != np.bool_ or not valid.any():
            raise ValueError("valid must be a boolean [468] mask with at least one supervised vertex")
        if not np.isfinite(vertices[valid]).all() or np.abs(vertices[valid]).max() > 10:
            raise ValueError("Supervised vertices must be finite relative coordinates, not millimeters")
        if valid.all() and not np.allclose(vertices.mean(axis=0), 0, atol=1e-4):
            raise ValueError("Complete targets must be centered before import")
        vertices = np.where(valid[:, None], vertices, 0).astype(np.float32) * scale
        return {"image": image, "vertices": torch.from_numpy(vertices), "valid": torch.from_numpy(valid.copy())}


def make_datasets(config):
    data, size = config.data, config.model.image_size
    if data.kind == "synthetic":
        return {split: SyntheticDataset(count, size, config.seed + offset)
                for split, count, offset in (("train", data.train_samples, 0), ("val", data.val_samples, 100000),
                                             ("test", data.test_samples, 200000))}
    datasets = {"train": ManifestDataset(data.root, data.train_manifest, size),
                "val": ManifestDataset(data.root, data.val_manifest, size)}
    if data.test_manifest:
        datasets["test"] = ManifestDataset(data.root, data.test_manifest, size)
    names = list(datasets)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            if (datasets[name].groups & datasets[other].groups
                    or datasets[name].image_paths & datasets[other].image_paths
                    or datasets[name].image_hashes & datasets[other].image_hashes):
                raise ValueError(f"Dataset split leakage between {name} and {other}: shared group or image content")
    return datasets
