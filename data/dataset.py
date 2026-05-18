"""
Dataset Manager — prepara il dataset per il training YOLO.

Operazioni:
  1. Split train/val/test (80/10/10) dei frame raccolti
  2. Augmentation via albumentations (flip, brightness, blur, noise)
  3. Generazione automatica di overwatch.yaml
  4. (Opzionale) merge con dataset Roboflow esterni

Uso:
  python data/dataset.py --augment --factor 3
"""

import os
import sys
import shutil
import random
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT        = Path(__file__).parent.parent
DATASET_DIR = ROOT / "datasets"
RAW_IMAGES  = DATASET_DIR / "images" / "raw"
RAW_LABELS  = DATASET_DIR / "labels" / "raw"

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}

CLASSES = {0: "enemy"}   # aggiungi classi se annoti anche allies, ecc.


# ─── Augmentation (implementata manualmente per non dipendere da albumentations) ─

def augment_image(image: np.ndarray, label_lines: List[str]) -> List[Tuple]:
    """
    Genera versioni aumentate di un singolo frame.
    Ritorna lista di (augmented_image, augmented_labels).
    """
    results = []

    # 1. Flip orizzontale
    flipped = cv2.flip(image, 1)
    flipped_labels = []
    for line in label_lines:
        parts = line.strip().split()
        cls, cx, cy, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        new_cx = 1.0 - cx   # specchia l'asse X
        flipped_labels.append(f"{cls} {new_cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    results.append((flipped, flipped_labels))

    # 2. Brightness jitter
    factor = random.uniform(0.6, 1.4)
    bright = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    results.append((bright, label_lines))

    # 3. Gaussian blur leggero
    ksize = random.choice([3, 5])
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    results.append((blurred, label_lines))

    # 4. Gaussian noise
    noise = np.random.normal(0, 8, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    results.append((noisy, label_lines))

    return results


# ─── Split ────────────────────────────────────────────────────────────────────

def split_dataset(seed: int = 42) -> Dict[str, list]:
    """Divide i file raw in train/val/test in modo riproducibile."""
    images = sorted(RAW_IMAGES.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"Nessuna immagine in {RAW_IMAGES}. Avvia prima collector.py")

    random.seed(seed)
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    return {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }


def copy_split(split_map: Dict[str, list], augment: bool = False,
               augment_factor: int = 3) -> None:
    """Copia (e opzionalmente augmenta) i file nelle cartelle split."""
    for split, paths in split_map.items():
        img_out = DATASET_DIR / "images" / split
        lbl_out = DATASET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[Dataset] {split}: {len(paths)} file")
        counter = 0

        for img_path in tqdm(paths, desc=split):
            lbl_path = RAW_LABELS / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Copia originale
            dst_img = img_out / f"{split}_{counter:06d}.jpg"
            dst_lbl = lbl_out / f"{split}_{counter:06d}.txt"
            shutil.copy(img_path, dst_img)
            shutil.copy(lbl_path, dst_lbl)
            counter += 1

            # Augmentation solo su train
            if augment and split == "train":
                image = cv2.imread(str(img_path))
                with open(lbl_path) as f:
                    label_lines = f.readlines()

                aug_samples = augment_image(image, [l.strip() for l in label_lines])
                # prendi solo augment_factor campioni casuali
                random.shuffle(aug_samples)
                for aug_img, aug_lbls in aug_samples[:augment_factor]:
                    dst_img = img_out / f"{split}_{counter:06d}.jpg"
                    dst_lbl = lbl_out / f"{split}_{counter:06d}.txt"
                    cv2.imwrite(str(dst_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    with open(dst_lbl, "w") as f:
                        f.write("\n".join(aug_lbls))
                    counter += 1

        print(f"  -> {counter} sample totali in {split}")


# ─── YAML ─────────────────────────────────────────────────────────────────────

def generate_yaml() -> Path:
    """Genera overwatch.yaml compatibile con Ultralytics YOLO."""
    yaml_path = ROOT / "data" / "overwatch.yaml"
    config = {
        "path":  str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASSES),
        "names": {int(k): v for k, v in CLASSES.items()},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n[Dataset] YAML salvato: {yaml_path}")
    return yaml_path


# ─── Stats ────────────────────────────────────────────────────────────────────

def print_stats() -> None:
    for split in ["train", "val", "test"]:
        imgs = list((DATASET_DIR / "images" / split).glob("*.jpg"))
        lbls = list((DATASET_DIR / "labels" / split).glob("*.txt"))
        total_boxes = sum(
            len(open(l).readlines()) for l in lbls if open(l).read().strip()
        )
        print(f"  {split:5s}: {len(imgs):5d} immagini  {total_boxes:6d} box totali")


# ─── Main ─────────────────────────────────────────────────────────────────────

def prepare(augment: bool = True, augment_factor: int = 3, seed: int = 42) -> Path:
    print("[Dataset] Split del dataset...")
    split_map = split_dataset(seed=seed)
    copy_split(split_map, augment=augment, augment_factor=augment_factor)
    yaml_path = generate_yaml()
    print("\n[Dataset] Statistiche finali:")
    print_stats()
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("--augment",  action="store_true", help="Applica augmentation")
    parser.add_argument("--factor",   type=int, default=3, help="Campioni aug per frame")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    prepare(augment=args.augment, augment_factor=args.factor, seed=args.seed)
