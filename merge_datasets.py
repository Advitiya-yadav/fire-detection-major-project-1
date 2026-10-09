import csv
import hashlib
import random
import shutil
from pathlib import Path

# ... [Keep your SOURCES, SMOKE_MODE, SPLIT_RATIOS, OUT_DIR definitions identical] ...

random.seed(42)

SOURCES = [
    {
        "name": "DF",
        "root": "Dataset/DF",
        "fire_dirs": ["Fire"],
        "nonfire_dirs": ["Non-Fire"],
        "is_aerial": True,   # your own drone-captured dataset
    },
    {
        "name": "ForestFireSmoke",
        "root": "data/raw/ForestFireSmoke",
        # Kaggle "Forest_Fire_Smoke_and_Non_Fire_Image_Dataset" (amerzishminha)
        # Ground-level imagery, 3 classes: fire, non fire, Smoke.
        "fire_dirs": ["train/fire", "test/fire"],
        "nonfire_dirs": ["train/non fire", "test/non fire"],
        "smoke_dirs": ["train/Smoke", "test/Smoke"],
        "is_aerial": False,   # ground-level photos, NOT drone footage
    },

  
]

# Smoke is kept as its own third class, in every source that has it.
SMOKE_MODE = "separate_class"   # "separate_class" | "merge_into_fire" | "drop"

# Same ratios for every source/class -- this is what guarantees smoke (and
# every other class) shows up proportionally in train/val/test, instead of
# test shrinking to whatever your smallest/aerial-only source can spare.
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Renamed from "dataset" -> "merged_dataset": on Windows/macOS, filesystems
# are case-insensitive by default, so a folder literally named "dataset"
# is the SAME folder as "Dataset/" (which holds the raw DF source). The old
# name meant merged output was being written straight into your raw data
# folder. This name doesn't collide, case-insensitively, with anything else
# in the project root.
OUT_DIR = Path("merged_dataset")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CHUNK_SIZE = 1 << 20  # 1 MiB, for hashing file content in chunks


def collect_images(root, subdirs):
    files = []
    for sd in subdirs:
        p = Path(root) / sd
        if not p.exists():
            print(f"  [skip] {p} does not exist -- check your SOURCES config")
            continue
        files.extend([f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS])
    return files


def file_content_hash(path):
    """MD5 of file *bytes* (not path) -- used to detect true duplicate
    images, even if they have different filenames or live in different
    source folders."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_by_content(files, seen_hashes, class_name):
    """Drop any file whose content hash we've already seen (globally,
    across all sources/classes). Returns the kept files. Mutates
    seen_hashes in place so duplicates across different classes are
    also caught (e.g. the same photo mislabeled in two source folders)."""
    kept = []
    dropped = 0
    for f in files:
        h = file_content_hash(f)
        if h in seen_hashes:
            dropped += 1
            continue
        seen_hashes[h] = (class_name, f)
        kept.append(f)
    if dropped:
        print(f"  [dedupe] {class_name}: dropped {dropped} duplicate-content "
              f"image(s) already seen elsewhere in the merge")
    return kept


def split_files(files, ratios):
    files = files.copy()
    random.shuffle(files)
    n = len(files)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:] if ratios["test"] > 0 else [],
    }


def copy_split(files_by_split, class_name, source_name, is_aerial, manifest_rows):
    for split, files in files_by_split.items():
        dest_dir = OUT_DIR / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            # Hash the full original path (not just filename) so files that
            # share a generic filename across different source subfolders
            # (very common in Kaggle datasets: 0001.jpg in both train/ and
            # test/) never collide on the destination name and silently
            # overwrite one another.
            path_hash = hashlib.md5(str(f.resolve()).encode()).hexdigest()[:8]
            dest_name = f"{source_name}_{path_hash}_{f.name}"
            dest_path = dest_dir / dest_name
            if dest_path.exists():
                print(f"  [warn] unexpected filename collision, skipping: {dest_path}")
                continue
            with open(f, "rb") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            manifest_rows.append({
                "dest_path": str(dest_path.relative_to(OUT_DIR)),
                "class": class_name,
                "split": split,
                "source": source_name,
                "is_aerial": is_aerial,
                "orig_path": str(f),
            })


def main():
    if SMOKE_MODE not in ("separate_class", "merge_into_fire", "drop"):
        raise ValueError(f"Unknown SMOKE_MODE: {SMOKE_MODE!r}")
    if OUT_DIR.exists():
        print(f"[INFO] Removing stale files from {OUT_DIR.resolve()}...")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    seen_hashes = {}      # content_hash -> (class_name, path), for dedup
    manifest_rows = []    # one row per file actually copied
    summary = []

    for src in SOURCES:
        name, is_aerial = src["name"], src.get("is_aerial", True)
        print(f"\nProcessing source: {name} ({'aerial' if is_aerial else 'ground-level'})")

        fire_files = collect_images(src["root"], src["fire_dirs"])
        nonfire_files = collect_images(src["root"], src["nonfire_dirs"])
        smoke_dirs = src.get("smoke_dirs", [])
        smoke_files = collect_images(src["root"], smoke_dirs) if smoke_dirs else []

        if smoke_files:
            if SMOKE_MODE == "merge_into_fire":
                print(f"  {len(smoke_files)} smoke images -> merged into fire class")
                fire_files += smoke_files
                smoke_files = []
            elif SMOKE_MODE == "drop":
                print(f"  {len(smoke_files)} smoke images -> dropped (SMOKE_MODE='drop')")
                smoke_files = []
            else:
                print(f"  {len(smoke_files)} smoke images -> kept as separate 'smoke' class")

        print(f"  found {len(fire_files)} fire, {len(nonfire_files)} nonfire, "
              f"{len(smoke_files)} smoke images (before dedup)")

        # Content-based dedup BEFORE splitting -- this is what prevents the
        # same physical photo from ending up in both train and test.
        fire_files = dedupe_by_content(fire_files, seen_hashes, "fire")
        nonfire_files = dedupe_by_content(nonfire_files, seen_hashes, "nonfire")
        smoke_files = dedupe_by_content(smoke_files, seen_hashes, "smoke") if smoke_files else []

        copy_split(split_files(fire_files, SPLIT_RATIOS), "fire", name, is_aerial, manifest_rows)
        copy_split(split_files(nonfire_files, SPLIT_RATIOS), "nonfire", name, is_aerial, manifest_rows)
        if smoke_files:
            copy_split(split_files(smoke_files, SPLIT_RATIOS), "smoke", name, is_aerial, manifest_rows)

        summary.append((name, len(fire_files), len(nonfire_files), len(smoke_files)))

    # ---- manifest.csv: per-file record of class/split/source/domain ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "dest_path", "class", "split", "source", "is_aerial", "orig_path"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # ---- summary ----
    print("\n=== Merge summary (post-dedup) ===")
    total_fire = total_nonfire = total_smoke = 0
    for name, nf, nn, ns in summary:
        print(f"  {name}: {nf} fire, {nn} nonfire, {ns} smoke")
        total_fire += nf
        total_nonfire += nn
        total_smoke += ns
    grand_total = total_fire + total_nonfire + total_smoke
    print(f"  TOTAL: {total_fire} fire, {total_nonfire} nonfire, {total_smoke} smoke "
          f"({grand_total} images combined)")
    print(f"\nEvery source uses the same {SPLIT_RATIOS} split, so fire/nonfire/smoke "
          f"all appear proportionally in train/val/test. To evaluate drone-only "
          f"performance specifically, filter manifest.csv where is_aerial == True "
          f"and split == 'test' -- no need to re-run the merge for that.")
    print(f"\nMerged dataset written to: {OUT_DIR.resolve()}")
    print(f"Manifest written to: {manifest_path.resolve()}")

    # ---- on-disk verification: does what's on disk match what we counted? ----
    print("\n=== On-disk verification ===")
    disk_total = 0
    for split in ("train", "val", "test"):
        for class_name in ("fire", "nonfire", "smoke"):
            d = OUT_DIR / split / class_name
            n = len(list(d.glob("*"))) if d.exists() else 0
            disk_total += n
            print(f"  {split}/{class_name}: {n}")
    print(f"  disk total: {disk_total}  |  expected total: {grand_total}")
    if disk_total != grand_total:
        print("  [warn] MISMATCH -- some files were not copied. "
              "Check for [warn]/[skip] lines above.")
    else:
        print("  OK -- disk matches expected count exactly.")


if __name__ == "__main__":
    main()