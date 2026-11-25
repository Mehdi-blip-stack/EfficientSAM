"""Download a reproducible COCO 2017 image subset with original annotations.

Only selected JPEGs are downloaded; full instance JSON files come from the
official annotation archive. No image resizing or mask modification occurs.
Run again after interruption: verified images and completed archives are reused.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import urllib.request
import zipfile

from PIL import Image

# Official COCO GCS bucket, also documented on the COCO downloads page.
# This HTTPS endpoint avoids a certificate-name mismatch on the custom domain.
BASE_URL = 'https://storage.googleapis.com/images.cocodataset.org'
ANNOTATIONS_URL = f'{BASE_URL}/annotations/annotations_trainval2017.zip'

def fetch(url, target):
    """Use a temporary file so interrupted downloads cannot appear complete."""
    temporary = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'EfficientSAM-course-project/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(target)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def verify_image(path, metadata):
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (metadata['width'], metadata['height'])
    except (OSError, ValueError):
        return False

def get_image(root, split, metadata):
    target = root / 'images' / split / metadata['file_name']
    if not verify_image(target, metadata):
        fetch(f"{BASE_URL}/{split}/{metadata['file_name']}", target)
    if not verify_image(target, metadata):
        raise ValueError(f'Image cannot be decoded or has wrong dimensions: {target}')
    return metadata['id']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=400)
    parser.add_argument('--val-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=36219)
    args = parser.parse_args()
    root = args.root.resolve()
    (root / 'annotations').mkdir(parents=True, exist_ok=True)
    (root / 'downloads').mkdir(exist_ok=True)
    archive = root / 'downloads/annotations_trainval2017.zip'
    annotation_paths = [root / f'annotations/instances_{split}.json' for split in ('train2017', 'val2017')]
    if not all(path.exists() for path in annotation_paths):
        if not archive.exists():
            print('Downloading official annotation archive (approximately 241 MB)...', flush=True)
            fetch(ANNOTATIONS_URL, archive)
        with zipfile.ZipFile(archive) as source:
            # Extract only the two instance files, not captions/keypoints.
            for path in annotation_paths:
                member = f'annotations/{path.name}'
                with source.open(member) as incoming, path.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    manifest = {'dataset': 'COCO 2017', 'seed': args.seed,
                'selection': 'uniform random image sampling without replacement, sorted IDs before sampling',
                'purpose': 'course-scale pilot; not a reproduction of full COCO benchmark scores',
                'validation_policy': 'held-out evaluation pilot; reserve a separate development split before later tuning',
                'source': 'https://cocodataset.org/#download', 'splits': {}}
    failures = []
    for split, count in [('train2017', args.train_count), ('val2017', args.val_count)]:
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        source_path = root / f'annotations/instances_{split}.json'
        print(f'Reading {source_path.name}...', flush=True)
        data = json.loads(source_path.read_text(encoding='utf-8'))
        # Separate random generators make either split reproducible independently.
        rng = random.Random(f'{args.seed}:{split}')
        selected = sorted(rng.sample(sorted(data['images'], key=lambda row: row['id']), count), key=lambda row: row['id'])
        ids = {row['id'] for row in selected}
        annotations = [row for row in data['annotations'] if row['image_id'] in ids]
        subset = {key: data[key] for key in ('info', 'licenses', 'categories') if key in data}
        subset.update(images=selected, annotations=annotations)
        (root / f'annotations/subset_instances_{split}.json').write_text(json.dumps(subset), encoding='utf-8')
        instance_counts = Counter(row['category_id'] for row in annotations)
        names = {row['id']: row['name'] for row in data['categories']}
        manifest['splits'][split] = {
            'image_ids': sorted(ids), 'image_count': len(selected), 'instance_count': len(annotations),
            'crowd_instances': sum(row.get('iscrowd', 0) for row in annotations),
            'observed_categories': len(instance_counts),
            'instances_per_category': {names[key]: value for key, value in sorted(instance_counts.items())},
            'source_annotation_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
        del data
        print(f'{split}: selected {len(selected)} images, {len(annotations)} instances; downloading JPEGs...', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(get_image, root, split, row): row for row in selected}
            for index, job in enumerate(as_completed(jobs), 1):
                try:
                    job.result()
                except Exception as error:
                    failures.append({'split': split, 'image_id': jobs[job]['id'], 'error': str(error)})
                if index % 25 == 0 or index == len(jobs):
                    print(f'{split}: checked {index}/{len(jobs)} images', flush=True)
    manifest['download_failures'] = failures
    manifest['complete'] = not failures
    (root / 'subset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} images failed. Rerun the same command to retry.')
    print(f'COMPLETE: verified {args.train_count + args.val_count} images. Manifest: {root / "subset_manifest.json"}', flush=True)

if __name__ == '__main__':
    main()
