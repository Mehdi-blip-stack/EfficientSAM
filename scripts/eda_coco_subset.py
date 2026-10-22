import json
from pathlib import Path
from collections import Counter
import os

DATA_ROOT = Path(os.environ["COCO_DATA_ROOT"])

# Read only our smaller training annotation file.
annotation_file = (
    DATA_ROOT / "annotations" / "subset_instances_train2017.json"
)
with annotation_file.open(encoding="utf-8") as file:
    data = json.load(file)

# Connect category numbers to names, such as "person".
category_names = {
    category["id"]: category["name"]
    for category in data["categories"]
}

# Count labelled objects in each category.
object_counts = Counter(
    category_names[annotation["category_id"]]
    for annotation in data["annotations"]
)
print("Ten most common object categories:")
for name, count in object_counts.most_common(10):
    print(f"{name}: {count}")


# Check all 80 categories, including those with no annotations.
all_counts = {
    category["name"]: object_counts.get(category["name"], 0)
    for category in data["categories"]
}

print("\nTen least common categories:")

for name, count in sorted(
    all_counts.items(),
    key=lambda item: item[1]
)[:10]:
    print(f"{name}: {count}")



# Read the annotations for our 200 evaluation photos.
val_file = (
    DATA_ROOT / "annotations" / "subset_instances_val2017.json"
)

with val_file.open(encoding="utf-8") as file:
    val_data = json.load(file)

# Count annotation entries for each category.
val_counts = Counter(
    category_names[annotation["category_id"]]
    for annotation in val_data["annotations"]
)

# Find categories with no evaluation annotations.
missing_categories = [
    category["name"]
    for category in val_data["categories"]
    if val_counts[category["name"]] == 0
]

print("\nEvaluation categories present:", len(val_counts))
print("Categories with no evaluation annotations:", missing_categories)



# Collect the width and height recorded for each training photo.
widths = [image["width"] for image in data["images"]]
heights = [image["height"] for image in data["images"]]

print("\nTraining photo dimensions:")
print("Width range:", min(widths), "to", max(widths), "pixels")
print("Height range:", min(heights), "to", max(heights), "pixels")



aspect_ratios = [
    image["width"] / image["height"]
    for image in data["images"]
]

print("\nAspect ratio range:")
print("Smallest:", round(min(aspect_ratios), 2))
print("Largest:", round(max(aspect_ratios), 2))



import matplotlib.pyplot as plt

# Group similar aspect ratios and count the photos in each group.
plt.hist(aspect_ratios, bins=20, edgecolor="black")

# Mark where square photos fall.
plt.axvline(
    1,
    color="red",
    linestyle="--",
    label="Square: ratio = 1"
)

plt.xlabel("Aspect ratio: width divided by height")
plt.ylabel("Number of training photos")
plt.title("Shapes of our 800 training photos")
plt.legend()
plt.tight_layout()
plt.show()


# Pick the first object annotation.
example = data["annotations"][0]

print("\nOne object annotation:")
print("Category:", category_names[example["category_id"]])
print("Bounding box:", example["bbox"])
print("Mask area:", example["area"], "pixels")


# Find the photo that this dog annotation belongs to.
example_image = next(
    image
    for image in data["images"]
    if image["id"] == example["image_id"]
)

image_area = example_image["width"] * example_image["height"]
coverage = example["area"] / image_area * 100

print("Photo size:", example_image["width"], "x", example_image["height"])
print("Dog coverage:", round(coverage, 2), "%")



# Quickly find each photo using its ID.
images_by_id = {
    image["id"]: image
    for image in data["images"]
}

coverages = []

for annotation in data["annotations"]:
    photo = images_by_id[annotation["image_id"]]
    photo_area = photo["width"] * photo["height"]

    percent = annotation["area"] / photo_area * 100
    coverages.append(percent)

print("\nObject coverage across training annotations:")
print("Smallest:", round(min(coverages), 4), "%")
print("Largest:", round(max(coverages), 2), "%")


# Find the annotation with the largest calculated coverage.
largest_index = max(
    range(len(coverages)),
    key=lambda index: coverages[index]
)

largest_annotation = data["annotations"][largest_index]
largest_photo = images_by_id[largest_annotation["image_id"]]

print("\nLargest-coverage annotation:")
print("Annotation ID:", largest_annotation["id"])
print("Category:", category_names[largest_annotation["category_id"]])
print("Image ID:", largest_photo["id"])
print("Photo size:", largest_photo["width"], "x", largest_photo["height"])
print("Stored area:", largest_annotation["area"])
print("Crowd flag:", largest_annotation["iscrowd"])
print("Segmentation format:", type(largest_annotation["segmentation"]).__name__)


polygons = largest_annotation["segmentation"]
print("\nPolygon checks:")
print("Number of polygons:", len(polygons))
x_values = [
    value
    for polygon in polygons
    for value in polygon[0::2]
]
y_values = [
    value
    for polygon in polygons
    for value in polygon[1::2]
]
print("X range:", min(x_values), "to", max(x_values))
print("Y range:", min(y_values), "to", max(y_values))


from pycocotools import mask as mask_utils

# Convert the truck's polygons into COCO's mask format.
polygon_masks = mask_utils.frPyObjects(
    largest_annotation["segmentation"],
    largest_photo["height"],
    largest_photo["width"]
)

# Combine the polygons. Count overlapping pixels only once.
combined_mask = mask_utils.merge(polygon_masks)

# Count foreground pixels in the combined mask.
pixel_area = int(mask_utils.area(combined_mask))

photo_area = largest_photo["width"] * largest_photo["height"]
pixel_coverage = pixel_area / photo_area * 100

print("\nTruck mask check:")
print("Combined mask area:", pixel_area, "pixels")
print("Combined mask coverage:", round(pixel_coverage, 2), "%")



from pycocotools.coco import COCO

# COCO's toolkit reads our saved subset.
coco = COCO(str(annotation_file))

mask_coverages = []
failed_masks = []

for annotation in data["annotations"]:
    photo = images_by_id[annotation["image_id"]]

    try:
        # Convert polygons or crowd annotations into a binary mask.
        mask = coco.annToMask(annotation)

        # Count foreground pixels, then calculate image coverage.
        coverage = (
            int(mask.sum())
            / (photo["width"] * photo["height"])
            * 100
        )

        mask_coverages.append(coverage)

    except Exception as error:
        # Record failures instead of silently ignoring them.
        failed_masks.append(
            (annotation["id"], str(error))
        )

print("\nDecoded mask results:")
print("Masks measured:", len(mask_coverages))
print("Masks that failed:", len(failed_masks))

if mask_coverages:
    print("Smallest coverage:", round(min(mask_coverages), 4), "%")
    print("Largest coverage:", round(max(mask_coverages), 2), "%")


import numpy as np

print(
    "Median mask coverage:",
    round(float(np.median(mask_coverages)), 2),
    "%"
)


plt.figure(figsize=(8, 5))
plt.hist(mask_coverages, bins=40, edgecolor="black")
plt.xlabel("Annotated region coverage (% of photo)")
plt.ylabel("Number of annotations")
plt.title("Training mask coverage")
plt.tight_layout()
plt.show()



# A coverage of zero means the decoded mask is empty.
empty_masks = sum(
    coverage == 0
    for coverage in mask_coverages
)
print("\nEmpty training masks:", empty_masks)



crowd_count = sum(
    annotation["iscrowd"] == 1
    for annotation in data["annotations"]
)
print("\nTraining crowd annotations:", crowd_count)



# Load our 200-image evaluation subset.
val_coco = COCO(str(val_file))
val_empty_masks = 0
val_failed_masks = []
for annotation in val_data["annotations"]:
    try:
        mask = val_coco.annToMask(annotation)
        if int(mask.sum()) == 0:
            val_empty_masks += 1
    except Exception as error:
        val_failed_masks.append(
            (annotation["id"], str(error))
        )
print("\nEvaluation mask checks:")
print("Annotations checked:", len(val_data["annotations"]))
print("Masks that failed:", len(val_failed_masks))
print("Empty masks:", val_empty_masks)



train_image_ids = {
    image["id"] for image in data["images"]
}
evaluation_image_ids = {
    image["id"] for image in val_data["images"]
}
shared_ids = train_image_ids & evaluation_image_ids
print("\nPhoto IDs shared between splits:", len(shared_ids))



# Find the first annotation marked as a crowd.
crowd_example = next(
    annotation
    for annotation in data["annotations"]
    if annotation["iscrowd"] == 1
)
crowd_photo = images_by_id[crowd_example["image_id"]]
print("\nCrowd example:")
print("Photo filename:", crowd_photo["file_name"])
print("Category:", category_names[crowd_example["category_id"]])



from PIL import Image
crowd_path = (
    DATA_ROOT / "images" / "train2017"
    / crowd_photo["file_name"]
)
crowd_mask = coco.annToMask(crowd_example)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
with Image.open(crowd_path) as photo:
    axes[0].imshow(photo.convert("RGB"))
axes[0].set_title("Original photo")
axes[1].imshow(crowd_mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("COCO chair crowd mask")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()


small_example = None
for annotation in data["annotations"]:
    if annotation["iscrowd"] == 1:
        continue
    photo = images_by_id[annotation["image_id"]]
    mask = coco.annToMask(annotation)
    coverage = (
        int(mask.sum())
        / (photo["width"] * photo["height"])
        * 100
    )
    if 0.5 <= coverage <= 1:
        small_example = annotation
        small_photo = photo
        print("\nSmall-object example:")
        print("Filename:", photo["file_name"])
        print("Category:", category_names[annotation["category_id"]])
        print("Coverage:", round(coverage, 2), "%")
        break


small_path = (
    DATA_ROOT / "images" / "train2017"
    / small_photo["file_name"]
)
small_mask = coco.annToMask(small_example)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
with Image.open(small_path) as photo:
    axes[0].imshow(photo.convert("RGB"))
axes[0].set_title("Original photo")
axes[1].imshow(small_mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Bicycle mask: 0.61% coverage")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()


# Read the full training annotations.
full_file = (
    DATA_ROOT / "annotations" / "instances_train2017.json"
)
with full_file.open(encoding="utf-8") as file:
    full_data = json.load(file)
# Count annotation entries for each category.
full_counts = Counter(
    category_names[annotation["category_id"]]
    for annotation in full_data["annotations"]
)
print("\nFull COCO training annotation count:")
print(sum(full_counts.values()))



full_total = sum(full_counts.values())
subset_total = sum(object_counts.values())
print("\nCategory | Full COCO % | Our subset %")
for name, count in full_counts.most_common(10):
    full_share = count / full_total * 100
    subset_share = object_counts.get(name, 0) / subset_total * 100
    print(f"{name}: {full_share:.2f}% | {subset_share:.2f}%")