from pathlib import Path
import shutil
from collections import defaultdict

# ---- Locate input_signatures relative to THIS script ----
BASE_DIR = Path(__file__).resolve().parent        # folder of this .py file
ROOT = BASE_DIR / "input_signatures"             # input_signatures is next to the script

print("BASE_DIR:", BASE_DIR)
print("ROOT:", ROOT, "exists?", ROOT.exists())

SETS = ["in_db", "not_in_db"]
valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# count how many images per signer (across both sets)
counters = defaultdict(int)

for set_name in SETS:
    set_dir = ROOT / set_name
    print(f"\n=== Processing set: {set_dir} ===")

    if not set_dir.exists():
        print(f"  [SKIP] {set_dir} does not exist")
        continue

    # find all images under in_db / not_in_db (recursively)
    imgs = [
        p for p in set_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    if not imgs:
        print(f"  [WARN] No images found in {set_dir}")
        continue

    for img in imgs:
        parent = img.parent.name  # e.g. "01" or "in_db"

        # signer id:
        if parent not in SETS:
            signer = parent                      # input_signatures/in_db/01/a.png -> "01"
        else:
            signer = img.stem.split("_")[0]      # input_signatures/in_db/01_a.png -> "01"

        counters[signer] += 1
        idx = counters[signer]

        new_name = f"{signer}_{idx}_{set_name}{img.suffix.lower()}"
        dst = ROOT / new_name

        print(f"  Moving {img} -> {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), str(dst))

    # clean up empty dirs under this set
    for sub in sorted(set_dir.glob("**/*"), reverse=True):
        if sub.is_dir():
            try:
                sub.rmdir()
                print(f"  Removed empty folder {sub}")
            except OSError:
                pass

    try:
        set_dir.rmdir()
        print(f"  Removed empty folder {set_dir}")
    except OSError:
        pass

print("\nDone.")
