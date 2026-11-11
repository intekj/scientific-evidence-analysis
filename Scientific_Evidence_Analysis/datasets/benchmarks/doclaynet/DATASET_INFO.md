# DocLayNet

## Purpose and official sources

DocLayNet is a human-annotated document-layout segmentation dataset covering
80,863 pages from six document categories: financial reports, scientific
articles, laws and regulations, government tenders, manuals, and patents. It
provides page images and COCO-format bounding-box annotations for 11 layout
classes.

- Official repository and dataset guide:
  [DS4SD/DocLayNet](https://github.com/DS4SD/DocLayNet)
- Hugging Face dataset:
  [ds4sd/DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet)
- Official core archive:
  [DocLayNet_core.zip](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip)

The official README reports the core archive as 28 GiB. It also offers a
separate 7.5 GiB extra archive containing page PDFs and text-cell JSON. Neither
full archive is downloaded by this project.

## Local sample and download method

The local subset is stored in `data/` as split-specific COCO JSON annotations
and only the matching PNG page images. It is selected from the official core
archive using HTTP byte-range requests, so the complete 28 GiB ZIP is not
downloaded or extracted. Sampling is deterministic (seed recorded in the
manifest) and balanced across the six official document categories. Original
COCO records, image identifiers, filenames, original split, and labels are
preserved without modifying annotation values.

The generated `data/manifest.json` records the source ETag, retrieval date,
requested page count, sample method, split counts, object counts, category
distribution, and annotation types. The downloaded split counts and storage
size are reported by `python inspect_multimodal_datasets.py`; these values are
not inferred from the requested sample size.

Prepared subset (downloaded 2026-11-10 UTC):

| Original split | Pages | Layout annotations |
|---|---:|---:|
| train | 50 | 771 |
| val (validation) | 50 | 758 |
| test | 50 | 745 |
| **Total** | **150** | **2,274** |

Each split contains 8–9 pages from each of the six official document
categories. The local subset occupies **63,491,380 bytes** including its
manifest and split annotation JSON files. Inspection validated all 150
image-to-annotation references, found 1025 × 1025 dimensions for every page,
and reported 0 missing or invalid records. The sampled annotation data contains
bounding boxes, segmentation polygons, area, crowd flags, precedence, and all
11 original class labels.

## Fields and annotations

Each selected image retains its original COCO and DocLayNet metadata, including
`id`, `file_name`, `width`, `height`, `doc_category`, `collection`, `doc_name`,
`page_no`, and `precedence` when present. Official page images are 1025 × 1025
PNG files.

The selected COCO annotation records retain the original `id`, `image_id`,
`category_id`, `bbox`, `segmentation`, `area`, `iscrowd`, and `precedence`
fields when present. Category IDs map to the original 11 layout labels in the
split's `categories` list. `image_id` associates every layout annotation with
its source page.

## Licence and use conditions

The official DocLayNet repository distributes its data under the Community
Data License Agreement – Permissive – Version 1.0 (CDLA-Permissive-1.0):
[official licence text](https://raw.githubusercontent.com/DS4SD/DocLayNet/main/LICENSE).
Review the agreement's conditions, attribution requirements, and any rights
applicable to source-document content before redistribution. The licence for
the dataset is distinct from licences for software used to process it.

## Relevance to this research

DocLayNet can support development and evaluation of document-layout-aware
evidence retrieval: detecting and matching text, tables, figures, captions, and
other page regions; retaining page and bounding-box provenance; and studying
whether multimodal context improves evidence alignment. It is a document
layout benchmark, not a scientific question-answering dataset and not a
flood-susceptibility dataset. The sample does not itself evaluate scientific
correctness or contradiction classification.

## Limitations

- This is a small, category-balanced development subset, not the full
  80,863-page benchmark. Results from it are not directly comparable to
  full-benchmark results.
- The subset keeps the original train/validation/test assignment, but small
  per-split samples have lower statistical power and may not represent every
  layout variation.
- Only core PNG and COCO data are selected. The extra PDFs and digital text-cell
  JSON are not downloaded.
- The images and annotation files are validated locally for matching IDs,
  dimensions, and referenced images. See `data/manifest.json` and the inspection
  script output for the actual sample counts and validation status.
