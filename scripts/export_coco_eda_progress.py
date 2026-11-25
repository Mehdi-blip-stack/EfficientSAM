"""Export completed COCO EDA findings. Chart and PDF use separate environments."""
import argparse
from collections import Counter
import json
import os
from statistics import median
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
ASSETS = ROOT / 'output/coco_eda'
DATA = Path(os.environ["COCO_DATA_ROOT"])

def chart():
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'from_scratch/runs/.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train = json.loads((DATA / 'annotations/subset_instances_train2017.json').read_text(encoding='utf-8'))
    names = {row['id']: row['name'] for row in train['categories']}
    counts = Counter(names[row['category_id']] for row in train['annotations'])
    evaluation = json.loads((DATA / 'annotations/subset_instances_val2017.json').read_text(encoding='utf-8'))
    evaluation_counts = Counter(row['category_id'] for row in evaluation['annotations'])
    from pycocotools.coco import COCO
    val_coco = COCO(str(DATA / 'annotations/subset_instances_val2017.json'))
    val_empty = 0
    val_failures = []
    for annotation in evaluation['annotations']:
        try:
            val_empty += int(val_coco.annToMask(annotation).sum()) == 0
        except Exception as error:
            val_failures.append({'annotation_id': annotation['id'], 'error': str(error)})
    top = counts.most_common(10)
    full = json.loads((DATA / 'annotations/instances_train2017.json').read_text(encoding='utf-8'))
    full_counts = Counter(names[row['category_id']] for row in full['annotations'])
    full_total = sum(full_counts.values())
    comparison = [{'category': name, 'full_percent': 100 * count / full_total,
                   'subset_percent': 100 * counts.get(name, 0) / len(train['annotations'])}
                  for name, count in full_counts.most_common(10)]
    del full
    ratios = [row['width'] / row['height'] for row in train['images']]
    example = train['annotations'][0]
    example_image = next(row for row in train['images'] if row['id'] == example['image_id'])
    from pycocotools import mask as mask_utils
    truck = next(row for row in train['annotations'] if row['id'] == 397929)
    truck_image = next(row for row in train['images'] if row['id'] == truck['image_id'])
    truck_masks = mask_utils.frPyObjects(truck['segmentation'], truck_image['height'], truck_image['width'])
    truck_union = int(mask_utils.area(mask_utils.merge(truck_masks)))
    from pycocotools.coco import COCO
    coco = COCO(str(DATA / 'annotations/subset_instances_train2017.json'))
    decoded_coverages = []
    decoding_failures = []
    for annotation in train['annotations']:
        photo_info = coco.imgs[annotation['image_id']]
        try:
            mask = coco.annToMask(annotation)
            decoded_coverages.append(100 * int(mask.sum()) / (photo_info['width'] * photo_info['height']))
        except Exception as error:
            decoding_failures.append({'annotation_id': annotation['id'], 'error': str(error)})
    all_counts = {name: counts.get(name, 0) for name in names.values()}
    crowd = next(a for a in train['annotations'] if a['iscrowd'] == 1)
    small = next(a for a, coverage in zip(train['annotations'], decoded_coverages)
                 if a['iscrowd'] == 0 and 0.5 <= coverage <= 1)
    licenses = {row['id']: row for row in train.get('licenses', [])}
    visual_sources = []
    for selected in (example, crowd, small):
        photo_info = coco.imgs[selected['image_id']]
        visual_sources.append({'id': photo_info['id'], 'source': photo_info.get('flickr_url', ''),
                               'license': licenses.get(photo_info['license'], {}).get('url', '')})
    summary = {'category_comparison': comparison, 'full_training_annotations': full_total, 'visual_sources': visual_sources, 'images': len(train['images']), 'annotations': len(train['annotations']),
               'width_range': [min(row['width'] for row in train['images']), max(row['width'] for row in train['images'])],
               'height_range': [min(row['height'] for row in train['images']), max(row['height'] for row in train['images'])],
               'aspect_ratio_range': [min(ratios), max(ratios)],
               'photo_shapes': {'portrait': sum(r < 1 for r in ratios), 'square': sum(r == 1 for r in ratios), 'landscape': sum(r > 1 for r in ratios)},
               'coverage_example': {'category': names[example['category_id']], 'image_id': example_image['id'], 'file_name': example_image['file_name'], 'source': example_image.get('flickr_url', ''), 'width': example_image['width'], 'height': example_image['height'], 'annotation_area': example['area'], 'percent': 100 * example['area'] / (example_image['width'] * example_image['height'])},
               'truck_anomaly': {'annotation_id': truck['id'], 'image_id': truck_image['id'], 'source': truck_image.get('flickr_url', ''), 'stored_area': truck['area'], 'union_area': truck_union, 'polygon_areas': mask_utils.area(truck_masks).tolist()},
               'decoded_training_masks': {'count': len(decoded_coverages), 'failures': decoding_failures, 'empty_masks': sum(value == 0 for value in decoded_coverages), 'minimum_percent': min(decoded_coverages), 'maximum_percent': max(decoded_coverages), 'median_percent': median(decoded_coverages)},
               'observed_categories': len(counts), 'top_categories': top,
               'rarest_categories': sorted(all_counts.items(), key=lambda item: item[1])[:10],
               'absent_categories': [name for name, count in all_counts.items() if count == 0],
               'evaluation_observed_categories': len(evaluation_counts),
               'shared_image_ids': sorted({row['id'] for row in train['images']} & {row['id'] for row in evaluation['images']}),
               'evaluation_mask_check': {'annotations': len(evaluation['annotations']), 'failures': val_failures, 'empty': val_empty},
               'evaluation_absent_categories': [row['name'] for row in evaluation['categories'] if evaluation_counts[row['id']] == 0],
               'person_share': counts['person'] / len(train['annotations']),
               'crowd_annotations': sum(row.get('iscrowd', 0) for row in train['annotations'])}
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / 'category_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels, values = zip(*reversed(top))
    ax.barh(labels, values, color='#237c9a')
    for y, value in enumerate(values):
        ax.text(value + 15, y, str(value), va='center', fontsize=9)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel('Annotation entries (not number of photos)')
    ax.set_title('COCO training subset: ten most common categories')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / 'category_counts.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    positions = list(range(len(comparison)))
    ax.bar([p-0.2 for p in positions], [row['full_percent'] for row in comparison], width=0.4, label='Full COCO training')
    ax.bar([p+0.2 for p in positions], [row['subset_percent'] for row in comparison], width=0.4, label='Our training subset')
    ax.set_xticks(positions, [row['category'] for row in comparison], rotation=30, ha='right')
    ax.set_ylabel('Share of annotation entries (%)')
    ax.set_title('Ten most frequent categories in full COCO training')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / 'category_comparison.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(ratios, bins=20, edgecolor='black', color='#237c9a')
    ax.axvline(1, color='red', linestyle='--', label='Square: ratio = 1')
    ax.set_xlabel('Aspect ratio: width divided by height')
    ax.set_ylabel('Number of training photos')
    ax.set_title('Shapes of our 800 training photos')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / 'aspect_ratios.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(decoded_coverages, bins=40, edgecolor='black', color='#237c9a')
    ax.set_xlabel('Annotated region coverage (% of photo)')
    ax.set_ylabel('Number of annotations')
    ax.set_title('Training mask coverage')
    fig.tight_layout()
    fig.savefig(ASSETS / 'mask_coverage.png', dpi=200)
    plt.close(fig)
    from PIL import Image as PILImage
    with PILImage.open(DATA / 'images/train2017' / truck_image['file_name']) as original:
        truck_photo = original.convert('RGB')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(truck_photo)
    axes[0].set_title('Original COCO image 475754')
    axes[1].imshow(mask_utils.decode(mask_utils.merge(truck_masks)), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Combined truck mask: 96.67%')
    for axis in axes:
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(ASSETS / 'truck_reference.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for row, (ann, title) in enumerate([(example, 'Large object: dog'),
                                       (crowd, 'Occlusion: chair crowd'),
                                       (small, 'Small object: bicycle')]):
        photo_info = coco.imgs[ann['image_id']]
        with PILImage.open(DATA / 'images/train2017' / photo_info['file_name']) as photo:
            axes[row, 0].imshow(photo.convert('RGB'))
        binary_mask = coco.annToMask(ann)
        percent = 100 * int(binary_mask.sum()) / binary_mask.size
        axes[row, 0].set_title(f"{title} | image {photo_info['id']}")
        axes[row, 1].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Annotation mask: {percent:.2f}% coverage')
        for axis in axes[row]:
            axis.axis('off')
    fig.tight_layout()
    fig.savefig(ASSETS / 'raw_examples.png', dpi=200)
    plt.close(fig)
    print(json.dumps(summary, indent=2))

def pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    summary = json.loads((ASSETS / 'category_summary.json').read_text())
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9.5
    styles['BodyText'].leading = 12
    styles['Heading2'].spaceBefore = 8
    styles['Heading2'].spaceAfter = 5
    styles['Heading2'].keepWithNext = True
    story = []
    def para(text, style='BodyText'):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 4))
    para('EfficientSAM: COCO Exploratory Data Analysis', 'Title')
    para('Working draft | Updated 22 October 2026 | Completed analyses only')
    para('1. Dataset and scope', 'Heading2')
    para('This course project uses official pretrained EfficientSAM checkpoints. '
         'The selected COCO 2017 pilot contains 800 images from the official training split '
         'and 200 from the official validation split, reserved for held-out evaluation. '
         'Sorted image metadata were sampled without replacement using Python random seeds '
         '36219 and 36220 respectively. This random pilot is not guaranteed to represent every '
         'category or reproduce full COCO benchmark results.')
    para('The selected training images contain 5,916 annotation entries; the evaluation images '
         'contain 1,642. The user-run download checks reported that all 1,000 images decoded '
         'successfully and matched the dimensions in the metadata. Mask decoding and empty-mask '
         'checks are reported below; comprehensive annotation-geometry checks remain pending.')
    para(f"The training and evaluation subsets share {len(summary['shared_image_ids'])} COCO image IDs. "
         'The two subsets are therefore disjoint by ID. Visual or near-duplicate similarity '
         'between different IDs has not been checked.')
    para('2. Training category distribution', 'Heading2')
    para(f"People account for 1,869 of 5,916 training annotation entries "
         f"({summary['person_share']:.1%}), compared with 313 car and 252 chair entries. "
         'The leading category therefore contributes substantially more annotations than '
         'the other categories shown. These are instance-annotation counts, not image counts; '
         'one image can contain several annotations of the same category.')
    story.append(Image(str(ASSETS / 'category_counts.png'), width=380, height=180.5))
    para(f"The subset contains annotations for {summary['observed_categories']} categories. "
         f"It includes {summary['crowd_annotations']} crowd annotation entries; these may describe "
         'groups rather than individual objects, so totals should not be interpreted as an exact '
         'count of individual physical objects.')
    story.append(Image(str(ASSETS / 'category_comparison.png'), width=380, height=171))
    para('Compared with 860,001 full training annotations, the subset broadly preserves the shares '
         'of these ten common categories. Person is 30.52% in full COCO versus 31.59% in the subset '
         '(+1.07 percentage points). This comparison does not establish rare-category coverage, '
         'representativeness of other visual attributes, or future model generalization.')
    story.append(PageBreak())
    para('3. Rare and absent training categories', 'Heading2')
    rare_text = '; '.join(f'{name}: {count}' for name, count in summary['rarest_categories'])
    para('The ten least frequent categories are ' + rare_text + '.')
    para('Toothbrush has no annotations in the training subset; toaster has only one. '
         'This describes annotation coverage, not proof that these objects are visually absent. '
         'Reliable category-specific conclusions cannot be drawn from such small counts. '
         'These coverage gaps must be considered when interpreting later evaluation results.')
    para('4. Evaluation category coverage', 'Heading2')
    absent_text = ', '.join(summary['evaluation_absent_categories'])
    para(f"The 200 held-out evaluation images contain annotations for {summary['evaluation_observed_categories']} "
         f"of the 80 categories. Categories without annotations are {absent_text}. "
         'Per-category mask-overlap performance cannot be measured for these categories on this subset. '
         'Future results must report the evaluated category coverage and must not be described as '
         'full 80-category COCO benchmark results.')
    val_check = summary['evaluation_mask_check']
    para(f"All {val_check['annotations']:,} evaluation annotation masks were checked: "
         f"{len(val_check['failures'])} decoding failures and {val_check['empty']} empty masks. "
         'This confirms technical mask readability, not semantic correctness of every annotation.')
    para('5. Training image dimensions', 'Heading2')
    para(f"Widths range from {summary['width_range'][0]} to {summary['width_range'][1]} pixels; "
         f"heights range from {summary['height_range'][0]} to {summary['height_range'][1]} pixels. "
         'These are separate ranges and need not describe the same image. '
         f"Width-to-height ratios range from {summary['aspect_ratio_range'][0]:.2f} "
         f"to {summary['aspect_ratio_range'][1]:.2f}, showing both portrait and landscape images. "
         'Direct square resizing can distort object shapes; later preprocessing must explicitly '
         'document its resizing policy and transform masks and prompt coordinates consistently.')
    story.append(Image(str(ASSETS / 'aspect_ratios.png'), width=440, height=209))
    shapes = summary['photo_shapes']
    para(f"The training subset contains {shapes['landscape']} landscape images (ratio above 1), "
         f"{shapes['portrait']} portrait images (below 1), and {shapes['square']} square images (equal to 1). "
         'Landscape images are most common. Histogram bars group nearby ratios; a bin crossing 1 '
         'should not be interpreted as an exact count of square images.')
    para('Planned evaluation metrics and their rationale', 'Heading3')
    para('For a predicted foreground mask P and ground-truth mask G, IoU = |P intersection G| / '
         '|P union G|, and Dice = 2|P intersection G| / (|P| + |G|). Both range from 0 (no overlap) '
         'to 1 (perfect overlap). Mean foreground IoU over eligible non-crowd object instances '
         'is the proposed primary score, with Dice as a companion. These related overlap scores '
         'do not constitute independent evidence. Pixel accuracy is unsuitable as the primary '
         'score because a mostly-background prediction can score highly while missing small objects.')
    para('Report point-prompt and box-prompt results separately under fixed prompt-generation rules. '
         'Include category and object-size breakdowns, their sample counts, and macro category '
         'averages over categories with eligible examples. Exclude crowd regions from the primary '
         'single-instance protocol and report their count. Subset overlap scores are not full COCO '
         'mask AP. These are proposed measurements, not model results; model evaluation has not begun.')
    story.append(PageBreak())
    para('6. Raw visual example and object coverage', 'Heading2')
    example = summary['coverage_example']
    story.append(Image(str(ASSETS / 'raw_examples.png'), width=480, height=432))
    para('Figure: Three deliberately selected training examples, with original photos on the left '
         'and decoded annotation masks on the right. White indicates foreground. These illustrate '
         'different challenges and are not a random sample of the dataset.')
    para(f"The first training annotation is a {example['category']} in a {example['width']} x "
         f"{example['height']} image. Its stored segmentation area is {example['annotation_area']:,.2f} "
         f"square pixels, covering {example['percent']:.2f}% of the image. "
         'Coverage is annotation area divided by image width times height, multiplied by 100. '
         'This is one illustrative object, not a dataset-wide summary. Stored polygon-based '
         'areas can be fractional and may differ slightly from decoded binary-mask pixel counts.')
    para('The chair crowd mask contains scattered visible regions in a scene with occlusion. '
         'The selected bicycle covers only 0.61% of its street photo, illustrating small-scale '
         'detail amid background clutter. The food-truck anomaly is shown on the next page.')
    for source in summary['visual_sources']:
        para(f"Image {source['id']} source: {source['source']}. License: {source['license']}")
    story.append(PageBreak())
    para('7. Area anomaly and mask verification', 'Heading2')
    story.append(Image(str(ASSETS / 'truck_reference.png'), width=400, height=200))
    para('Figure: Close-up of a food-truck service window (left) and the combined '
         'mask for its truck annotation (right). White pixels belong to the annotated region. '
         'The original photo is displayed without an overlay; the mask is a separate panel.')
    truck = summary['truck_anomaly']
    para(f"Truck annotation {truck['annotation_id']} in image {truck['image_id']} has stored area "
         f"{truck['stored_area']:,.2f} in a 500 x 500 image, implying 134.54% coverage. "
         'Its two polygons lie within image bounds but overlap. The official COCO mask toolkit '
         f"gives individual rasterized areas of {truck['polygon_areas'][0]:,} and {truck['polygon_areas'][1]:,} pixels; "
         f"their union is {truck['union_area']:,} pixels (96.67% coverage). "
         'The stored area closely matches the sum of the polygon areas rather than their union. '
         'For this example, overlapping area explains the inflated value. Subsequent coverage '
         'analysis uses decoded union masks; the training-wide check follows. '
         'Source annotations are preserved unchanged.')
    decoded = summary['decoded_training_masks']
    para('Training-wide mask check', 'Heading3')
    para(f"All {decoded['count']:,} training annotation entries were decoded with COCO annToMask; "
         f"{len(decoded['failures'])} failed, and {decoded['empty_masks']} decoded masks are empty. "
         'Each training annotation therefore yields at least one foreground pixel. '
         'Coverage measured from binary foreground pixels ranges '
         f"from {decoded['minimum_percent']:.4f}% to {decoded['maximum_percent']:.2f}%. "
         f"Median coverage is {decoded['median_percent']:.2f}%, placing the middle annotation below "
         '1% of image area and demonstrating that small relative regions are common. '
         'Thus the subset includes very small regions and regions occupying almost an entire image. '
         'Small regions may lose detail during downsampling. Successful decoding confirms that masks '
         'can be constructed; it does not establish that every annotation is semantically correct. '
         'These results cover training annotations only, including crowd entries.')
    story.append(Image(str(ASSETS / 'mask_coverage.png'), width=400, height=180))
    para('The histogram is right-skewed: most annotation regions occupy a small fraction '
         'of image area, with a long tail of larger regions. A bin near zero spans a range '
         'of small coverages; its height does not mean those masks have zero area.')
    para('Image source recorded in COCO: ' + truck['source'] + '. '
         'License: Attribution-NoDerivs 2.0 (https://creativecommons.org/licenses/by-nd/2.0/).')
    para('Source: COCO 2017 instance annotations, https://cocodataset.org/#download. '
         'Counts computed from saved subsets; image-check results supplied by the user.')
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'EfficientSAM_COCO_EDA_Draft.pdf'
    if isinstance(story[-1], Spacer):
        story.pop()
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#526174'))
        canvas.drawString(42, 22, 'COCO EDA | Working draft')
        canvas.drawRightString(A4[0]-42, 22, str(doc.page))
    SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=42, rightMargin=42,
                      topMargin=32, bottomMargin=36).build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['chart', 'pdf'])
    args = parser.parse_args()
    chart() if args.stage == 'chart' else pdf()
