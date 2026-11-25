import json
import random
import time
import urllib.request
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


# --------------------------------------------------
# 1. Locate the dataset
# --------------------------------------------------

# The folder name still says 500, but we select 1,000 images.
DATA_ROOT = Path(os.environ["COCO_DATA_ROOT"])

annotations_folder = DATA_ROOT / "annotations"
train_folder = DATA_ROOT / "images" / "train2017"
train_folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Read the original annotation files
# --------------------------------------------------

train_file = annotations_folder / "instances_train2017.json"

with train_file.open(encoding="utf-8") as file:
    train_data = json.load(file)

print("Training images available:", len(train_data["images"]))

val_file = annotations_folder / "instances_val2017.json"

with val_file.open(encoding="utf-8") as file:
    val_data = json.load(file)


# --------------------------------------------------
# 3. Select 800 training and 200 evaluation images
# --------------------------------------------------

# Sorting and fixed seeds make our selection repeatable.
train_images = sorted(
    train_data["images"],
    key=lambda image: image["id"]
)

train_rng = random.Random(36219)
selected_train = train_rng.sample(train_images, 800)

val_images = sorted(
    val_data["images"],
    key=lambda image: image["id"]
)

val_rng = random.Random(36220)
selected_val = val_rng.sample(val_images, 200)

print("Selected training images:", len(selected_train))
print("Selected evaluation images:", len(selected_val))


# --------------------------------------------------
# 4. Save the selected image IDs
# --------------------------------------------------

selection = {
    "train_seed": 36219,
    "evaluation_seed": 36220,
    "train_image_ids": [
        image["id"] for image in selected_train
    ],
    "evaluation_image_ids": [
        image["id"] for image in selected_val
    ],
}

selection_file = DATA_ROOT / "subset_selection_1000.json"

with selection_file.open("w", encoding="utf-8") as file:
    json.dump(selection, file, indent=2)

print("Selection saved to:", selection_file)


# --------------------------------------------------
# 5. Collect annotations for the selected images
# --------------------------------------------------

train_ids = set(selection["train_image_ids"])
val_ids = set(selection["evaluation_image_ids"])

selected_train_annotations = [
    annotation
    for annotation in train_data["annotations"]
    if annotation["image_id"] in train_ids
]

selected_val_annotations = [
    annotation
    for annotation in val_data["annotations"]
    if annotation["image_id"] in val_ids
]

print("Selected training objects:", len(selected_train_annotations))
print("Selected evaluation objects:", len(selected_val_annotations))


# --------------------------------------------------
# 6. Save smaller annotation files
# --------------------------------------------------

train_subset = {
    "images": selected_train,
    "annotations": selected_train_annotations,
    "categories": train_data["categories"],
}

val_subset = {
    "images": selected_val,
    "annotations": selected_val_annotations,
    "categories": val_data["categories"],
}

# Preserve the source information and image-license descriptions.
for key in ("info", "licenses"):
    if key in train_data:
        train_subset[key] = train_data[key]

    if key in val_data:
        val_subset[key] = val_data[key]

train_subset_file = (
    annotations_folder / "subset_instances_train2017.json"
)

with train_subset_file.open("w", encoding="utf-8") as file:
    json.dump(train_subset, file)

print("Training annotations saved.")

val_subset_file = (
    annotations_folder / "subset_instances_val2017.json"
)

with val_subset_file.open("w", encoding="utf-8") as file:
    json.dump(val_subset, file)

print("Evaluation annotations saved.")


# --------------------------------------------------
# 7. Define a photo-checking function
# --------------------------------------------------

def photo_is_valid(path, photo_info):
    """Return True if the photo opens and has the expected dimensions."""

    if not path.is_file():
        return False

    try:
        with Image.open(path) as photo:
            photo.load()

            expected_size = (
                photo_info["width"],
                photo_info["height"]
            )

            return photo.size == expected_size

    except (OSError, ValueError):
        return False


# --------------------------------------------------
# 8. Download the selected training photos
# --------------------------------------------------

for number, photo_info in enumerate(selected_train, start=1):
    photo_path = train_folder / photo_info["file_name"]

    # Keep photos that are already downloaded and readable.
    if photo_is_valid(photo_path, photo_info):
        continue

    photo_url = (
        "https://s3.amazonaws.com/images.cocodataset.org/"
        f"train2017/{photo_info['file_name']}"
    )

    # Keep incomplete downloads separate from finished JPEGs.
    temporary_path = photo_path.with_suffix(".jpg.part")

    # Try this photo up to five times before stopping.
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(
                photo_url, timeout=60
            ) as response:
                with temporary_path.open("wb") as file:
                    # Read small chunks instead of the whole photo at once.
                    while True:
                        chunk = response.read(64 * 1024)

                        if not chunk:
                            break

                        file.write(chunk)

            if not photo_is_valid(temporary_path, photo_info):
                raise ValueError(
                    "Downloaded photo is unreadable or has wrong dimensions"
                )

            # Rename only after the download passes our check.
            temporary_path.replace(photo_path)

            print(
                f"Training photo {number}/{len(selected_train)} saved",
                flush=True
            )

            break

        except (OSError, ValueError) as error:
            print(
                f"Attempt {attempt}/5 failed for photo {number}: {error}",
                flush=True
            )

            if attempt == 5:
                print(
                    "Download stopped. Run this script again to retry.",
                    flush=True
                )
                raise

            time.sleep(3)

print("Training photo download finished.")


# --------------------------------------------------
# 9. Check all 800 training photos
# --------------------------------------------------

problem_files = []

for photo_info in selected_train:
    photo_path = train_folder / photo_info["file_name"]

    if not photo_is_valid(photo_path, photo_info):
        problem_files.append(photo_info["file_name"])

print("Training photos checked:", len(selected_train))
print("Missing or damaged photos:", len(problem_files))

if problem_files:
    print("Files needing attention:", problem_files)


# --------------------------------------------------
# 10. Inspect the first photo and its labelled boxes
# --------------------------------------------------

image = selected_train[0]
save_path = train_folder / image["file_name"]

category_names = {
    category["id"]: category["name"]
    for category in train_data["categories"]
}

photo_annotations = [
    annotation
    for annotation in selected_train_annotations
    if annotation["image_id"] == image["id"]
]

photo_labels = [
    category_names[annotation["category_id"]]
    for annotation in photo_annotations
]

print("Labels for the first photo:", photo_labels)

with Image.open(save_path) as photo:
    # Convert for consistent display; the saved original stays unchanged.
    display_photo = photo.convert("RGB")

fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(display_photo)

for annotation in photo_annotations:
    x, y, width, height = annotation["bbox"]
    label = category_names[annotation["category_id"]]

    box = Rectangle(
        (x, y),
        width,
        height,
        fill=False,
        edgecolor="red",
        linewidth=2
    )

    ax.add_patch(box)

    ax.text(
        x,
        y,
        label,
        color="white",
        backgroundcolor="red"
    )

ax.set_title("First selected training photo: COCO annotation boxes")
ax.axis("off")

plt.tight_layout()
plt.show()


# Create a separate folder for evaluation photos.
val_folder = DATA_ROOT / "images" / "val2017"
val_folder.mkdir(parents=True, exist_ok=True)

for number, photo_info in enumerate(selected_val, start=1):
    photo_path = val_folder / photo_info["file_name"]

    # Keep existing photos if they pass our check.
    if photo_is_valid(photo_path, photo_info):
        continue

    photo_url = (
        "https://s3.amazonaws.com/images.cocodataset.org/"
        f"val2017/{photo_info['file_name']}"
    )

    temporary_path = photo_path.with_suffix(".jpg.part")

    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(
                photo_url, timeout=60
            ) as response:
                with temporary_path.open("wb") as file:
                    while True:
                        chunk = response.read(64 * 1024)

                        if not chunk:
                            break

                        file.write(chunk)

            if not photo_is_valid(temporary_path, photo_info):
                raise ValueError("Downloaded photo failed its check")

            temporary_path.replace(photo_path)

            print(
                f"Evaluation photo {number}/200 saved",
                flush=True
            )
            break

        except (OSError, ValueError) as error:
            print(f"Attempt {attempt}/5 failed: {error}")

            if attempt == 5:
                raise

            time.sleep(3)

# Check every selected evaluation photo.
val_problem_files = [
    photo_info["file_name"]
    for photo_info in selected_val
    if not photo_is_valid(
        val_folder / photo_info["file_name"],
        photo_info
    )
]

print("Evaluation photos checked:", len(selected_val))
print("Missing or damaged evaluation photos:", len(val_problem_files))