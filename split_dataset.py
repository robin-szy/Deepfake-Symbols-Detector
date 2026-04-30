"""
Split the original symbols dataset into:

symbols_split/
    train_val/
        real/
        fake/
    test/
        real/
        fake/
    test_flat/
        *.csv
    test_labels.csv
"""

from pathlib import Path
import random
import shutil
import csv


# ===== Config =====
SOURCE_DIR = Path("./symbols")          # original folder with /real and /fake
OUTPUT_DIR = Path("./symbols_split")    # new split folder
TEST_RATIO = 0.1
SEED = 42
COPY_FILES = True                       # True = copy, False = move

LABELS = {
    "real": 0,
    "fake": 1,
}


def reset_output_dir(path: Path):
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Delete it manually first if you want to recreate the split."
        )
    path.mkdir(parents=True)


def copy_or_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main():
    random.seed(SEED)
    reset_output_dir(OUTPUT_DIR)

    test_labels = []

    for class_name, label in LABELS.items():
        class_dir = SOURCE_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        files = sorted(class_dir.glob("*.csv"))
        random.shuffle(files)

        n_test = max(1, round(len(files) * TEST_RATIO))
        test_files = files[:n_test]
        train_val_files = files[n_test:]

        print(f"{class_name}: {len(train_val_files)} train_val, {len(test_files)} test")

        # train_val keeps labels in folder names
        for src in train_val_files:
            dst = OUTPUT_DIR / "train_val" / class_name / src.name
            copy_or_move(src, dst)

        # test also keeps labels, but should stay untouched
        for src in test_files:
            dst = OUTPUT_DIR / "test" / class_name / src.name
            copy_or_move(src, dst)

            # flat test folder simulates professor's prediction folder
            # Prefix filename with class to avoid collisions between real/fake files.
            flat_name = f"{class_name}_{src.name}"
            flat_dst = OUTPUT_DIR / "test_flat" / flat_name
            flat_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, flat_dst)

            test_labels.append((str(flat_dst.resolve()), label))

    # Save local test labels separately.
    labels_path = OUTPUT_DIR / "test_labels.csv"
    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "label"])
        writer.writerows(test_labels)

    print("\nDone.")
    print(f"Train/validation data: {OUTPUT_DIR / 'train_val'}")
    print(f"Untouched test data:   {OUTPUT_DIR / 'test'}")
    print(f"Flat eval folder:      {OUTPUT_DIR / 'test_flat'}")
    print(f"Local test labels:     {labels_path}")


if __name__ == "__main__":
    main()
