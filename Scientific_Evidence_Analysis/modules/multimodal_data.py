"""Small helpers for validating locally sampled multimodal benchmark records."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


def read_png_dimensions(path: str | Path) -> tuple[int, int]:
    """Read PNG dimensions from its header without decoding image pixels."""
    image_path = Path(path)
    with image_path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a complete PNG file: {image_path}")
    if header[12:16] != b"IHDR":
        raise ValueError(f"PNG has no IHDR header: {image_path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"PNG has invalid dimensions: {image_path}")
    return width, height


def validate_doclaynet_split(
    annotation_file: str | Path,
    image_root: str | Path,
) -> dict[str, Any]:
    """Validate COCO subset integrity and confirm every image/annotation match."""
    annotation_path = Path(annotation_file)
    images_dir = Path(image_root)
    with annotation_path.open("r", encoding="utf-8") as stream:
        coco = json.load(stream)
    if not isinstance(coco, dict):
        raise ValueError(f"COCO annotations must be a JSON object: {annotation_path}")

    image_records = coco.get("images", [])
    annotation_records = coco.get("annotations", [])
    if not isinstance(image_records, list) or not isinstance(annotation_records, list):
        raise ValueError(f"COCO image and annotation fields must be lists: {annotation_path}")

    image_ids: set[int] = set()
    invalid_records: list[str] = []
    image_dimensions: dict[str, list[int]] = {}
    for image in image_records:
        if not isinstance(image, dict):
            invalid_records.append("image record is not an object")
            continue
        image_id = image.get("id")
        filename = image.get("file_name")
        if not isinstance(image_id, int) or not isinstance(filename, str):
            invalid_records.append("image record missing integer id or file_name")
            continue
        if image_id in image_ids:
            invalid_records.append(f"duplicate image_id {image_id}")
        image_ids.add(image_id)
        if Path(filename).name != filename:
            invalid_records.append(f"unsafe image filename {filename!r}")
            continue
        image_path = images_dir / filename
        if not image_path.is_file():
            invalid_records.append(f"missing image {filename}")
            continue
        try:
            width, height = read_png_dimensions(image_path)
        except (OSError, ValueError) as exc:
            invalid_records.append(f"invalid image {filename}: {exc}")
            continue
        image_dimensions[filename] = [width, height]
        if (
            image.get("width") != width
            or image.get("height") != height
        ):
            invalid_records.append(
                f"dimension mismatch for {filename}: "
                f"annotation={image.get('width')}x{image.get('height')}, "
                f"image={width}x{height}"
            )

    annotation_ids: set[int] = set()
    annotation_types: set[str] = set()
    category_ids = {
        category.get("id")
        for category in coco.get("categories", [])
        if isinstance(category, dict)
    }
    for annotation in annotation_records:
        if not isinstance(annotation, dict):
            invalid_records.append("annotation record is not an object")
            continue
        annotation_id = annotation.get("id")
        if isinstance(annotation_id, int):
            if annotation_id in annotation_ids:
                invalid_records.append(f"duplicate annotation id {annotation_id}")
            annotation_ids.add(annotation_id)
        image_id = annotation.get("image_id")
        if image_id not in image_ids:
            invalid_records.append(
                f"annotation {annotation_id!r} references unknown image_id {image_id!r}"
            )
        category_id = annotation.get("category_id")
        if category_ids and category_id not in category_ids:
            invalid_records.append(
                f"annotation {annotation_id!r} references unknown category_id "
                f"{category_id!r}"
            )
        if isinstance(category_id, int):
            annotation_types.add(f"category_id:{category_id}")
        if isinstance(annotation.get("bbox"), list):
            annotation_types.add("bbox")
        if "segmentation" in annotation:
            annotation_types.add("segmentation")
        if "area" in annotation:
            annotation_types.add("area")
        if "iscrowd" in annotation:
            annotation_types.add("iscrowd")

    return {
        "record_count": len(image_records),
        "annotation_count": len(annotation_records),
        "matched_images": len(image_dimensions),
        "image_dimensions": image_dimensions,
        "annotation_types": sorted(annotation_types),
        "category_names": [
            item.get("name")
            for item in coco.get("categories", [])
            if isinstance(item, dict)
        ],
        "missing_or_invalid_records": invalid_records,
        "example_annotation": (
            annotation_records[0] if annotation_records else None
        ),
        "fields": sorted(
            {
                key
                for record in image_records + annotation_records
                if isinstance(record, dict)
                for key in record
            }
        ),
    }
