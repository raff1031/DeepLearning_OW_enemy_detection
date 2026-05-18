"""
Scarica dataset OW2 pubblici da Roboflow Universe.
Tenta piu' dataset in ordine di dimensione/qualita'.
"""
import os, sys, shutil, zipfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent

DATASETS = [
    {
        "name": "overwatch-bot (533 img)",
        "workspace": "overwatch2-xbudr",
        "project":   "overwatch-bot",
        "version":   1,
    },
    {
        "name": "overwatch-character-detection (1583 img)",
        "workspace": "appliedrobotics-uihpz",
        "project":   "overwatch-2-character-detection",
        "version":   1,
    },
    {
        "name": "overwatch-aimbot (240 img)",
        "workspace": "evox",
        "project":   "overwatch-aimbot",
        "version":   1,
    },
]

def try_roboflow(api_key: str) -> bool:
    from roboflow import Roboflow

    for ds in DATASETS:
        print(f"\n[Download] Provo: {ds['name']}")
        try:
            rf      = Roboflow(api_key=api_key)
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            dest    = str(ROOT / "datasets" / "roboflow_raw")
            dataset = version.download("yolov8", location=dest, overwrite=True)
            print(f"  OK! Salvato in: {dest}")
            return dest
        except Exception as e:
            print(f"  FAIL: {e}")
    return None


def merge_into_raw(roboflow_dir: str) -> int:
    """Copia immagini e label da un dataset Roboflow nel nostro formato raw."""
    src = Path(roboflow_dir)
    img_dst = ROOT / "datasets" / "images" / "raw"
    lbl_dst = ROOT / "datasets" / "labels" / "raw"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    existing = len(list(img_dst.glob("*.jpg")))

    for split in ["train", "valid", "test"]:
        img_src = src / split / "images"
        lbl_src = src / split / "labels"
        if not img_src.exists():
            continue
        for img_path in img_src.glob("*.jpg"):
            lbl_path = lbl_src / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            idx = existing + count
            shutil.copy(img_path, img_dst / f"{idx:06d}.jpg")
            shutil.copy(lbl_path, lbl_dst / f"{idx:06d}.txt")
            count += 1

    print(f"[Merge] Copiati {count} campioni in datasets/images/raw")
    return count


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else ""

    if not api_key:
        print("=" * 60)
        print("Serve una API key gratuita di Roboflow.")
        print("1. Vai su https://app.roboflow.com  (signup gratuito)")
        print("2. Copia la tua API key da Settings > Roboflow API")
        print("3. Rilancia: python download_dataset.py TUA_API_KEY")
        print("=" * 60)
        print("\nNel frattempo uso il dataset sintetico gia' presente.")
        sys.exit(1)

    dest = try_roboflow(api_key)
    if dest:
        n = merge_into_raw(dest)
        print(f"\nPronto! {n} immagini nel dataset raw.")
        print("Prossimo step: python main.py --mode dataset --augment")
    else:
        print("\nTutti i download falliti. Controlla la API key.")
