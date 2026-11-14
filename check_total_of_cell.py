
from pathlib import Path
from collections import Counter

dataset = Path("dataset_v3")
class_names = {0: "Platelet", 1: "RBC", 2: "WBC"}

total_counts = Counter()

for split in ["train", "valid", "test"]:
    counts = Counter()

    for file in (dataset / split / "labels").glob("*.txt"):
        for line in file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                counts[class_id] += 1

    total_counts.update(counts)

    print(f"\n{split.upper()}")
    for class_id, name in class_names.items():
        print(f"{name}: {counts[class_id]}")
    print(f"Total: {sum(counts.values())}")

print("\nALL DATASET")
for class_id, name in class_names.items():
    print(f"{name}: {total_counts[class_id]}")
print(f"Total: {sum(total_counts.values())}")
