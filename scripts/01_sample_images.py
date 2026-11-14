
from pathlib import Path
import random
import shutil
import csv

# =================================
# CONFIG
# =================================

RAW_DIR = Path("raw_dataset/images")

OUTPUT_DIR = Path("annotation_dataset/images")

REPORT_PATH = Path(
    "annotation_dataset/sampling_report.csv"
)

SAMPLE_SIZE = 150

RANDOM_SEED = 42

EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}

# =================================
# GET IMAGE LIST
# =================================

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Cannot find: {RAW_DIR}"
    )

images = sorted([
    p for p in RAW_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in EXTENSIONS
])

print("Total images:", len(images))

if len(images) < SAMPLE_SIZE:
    raise ValueError("Not enough images")

# =================================
# RANDOM SAMPLING
# =================================

random.seed(RANDOM_SEED)

selected = random.sample(images, SAMPLE_SIZE)

# =================================
# CREATE OUTPUT
# =================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        "Output folder is not empty. "
        "Please use an empty folder."
    )

# =================================
# COPY AND REPORT
# =================================

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample_name",
        "original_path"
    ])

    for i, source in enumerate(selected, 1):

        new_name = (
            f"sample_{i:05d}"
            f"{source.suffix.lower()}"
        )

        destination = OUTPUT_DIR / new_name

        shutil.copy2(source, destination)

        writer.writerow([
            new_name,
            str(source.resolve())
        ])

        print(f"{i}/{SAMPLE_SIZE}: {new_name}")

print("\nSampling completed!")
print("Output:", OUTPUT_DIR)
print("Report:", REPORT_PATH)
