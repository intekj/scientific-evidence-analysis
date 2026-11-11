import json
import struct
import zipfile
from pathlib import Path

import pytest

from download_multimodal_subsets import (
    DOC_CATEGORIES,
    JSONStreamReader,
    SubsetDownloadError,
    _extract_image_member,
    iter_json_array,
    select_representative_images,
)
from modules.multimodal_data import (
    read_png_dimensions,
    validate_doclaynet_split,
)


def write_png_header(path: Path, width: int = 1025, height: int = 1025) -> None:
    header = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
    )
    path.write_bytes(header)


def coco_fixture(tmp_path: Path, *, include_image: bool = True) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    filename = "page-1.png"
    if include_image:
        write_png_header(image_root / filename)
    coco = {
        "info": {"description": "offline test fixture"},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": filename,
                "width": 1025,
                "height": 1025,
                "doc_category": "scientific_articles",
                "page_no": 1,
            }
        ],
        "annotations": [
            {
                "id": 4,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1.0, 2.0, 30.0, 40.0],
                "segmentation": [[1.0, 2.0, 31.0, 2.0, 31.0, 42.0]],
                "area": 1200.0,
                "iscrowd": 0,
                "precedence": 0,
            }
        ],
        "categories": [{"id": 1, "name": "Caption"}],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(coco), encoding="utf-8")
    return annotation_path, image_root


def test_json_stream_reader_loads_nested_values_and_arrays_in_order():
    payload = {
        "info": {"description": "test"},
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [{"id": 7, "image_id": 2}],
        "categories": [{"id": 1, "name": "Text"}],
    }
    reader = JSONStreamReader(
        __import__("io").BytesIO(json.dumps(payload).encode("utf-8")),
        chunk_size=8,
    )

    fields = {}
    for key, value in reader.items():
        fields[key] = list(value) if key in {"images", "annotations", "categories"} else value

    assert fields["info"]["description"] == "test"
    assert [record["id"] for record in fields["images"]] == [1, 2]
    assert fields["annotations"][0]["image_id"] == 2
    assert fields["categories"][0]["name"] == "Text"


def test_array_parser_handles_large_record_over_chunk_boundaries():
    records = [{"text": "x" * 100}, {"text": "y" * 100}]
    stream = __import__("io").BytesIO(
        json.dumps({"images": records}).encode("utf-8")
    )

    actual = list(iter_json_array(stream, "images", chunk_size=17))

    assert actual == records


def test_image_annotations_match_and_split_file_is_loadable(tmp_path):
    annotations, image_root = coco_fixture(tmp_path)

    report = validate_doclaynet_split(annotations, image_root)

    assert report["record_count"] == 1
    assert report["annotation_count"] == 1
    assert report["matched_images"] == 1
    assert report["image_dimensions"]["page-1.png"] == [1025, 1025]
    assert "bbox" in report["annotation_types"]
    assert report["example_annotation"]["image_id"] == 1
    assert report["missing_or_invalid_records"] == []


def test_split_preservation_and_required_field_validation(tmp_path):
    annotation_path, image_root = coco_fixture(tmp_path)
    split_directory = tmp_path / "train"
    split_directory.mkdir()
    split_file = split_directory / "instances.json"
    split_file.write_bytes(annotation_path.read_bytes())

    report = validate_doclaynet_split(split_file, image_root)
    coco = json.loads(split_file.read_text(encoding="utf-8"))

    assert split_file.parent.name == "train"
    assert report["record_count"] == 1
    assert all(
        field in coco["images"][0]
        for field in ("id", "file_name", "doc_category", "page_no")
    )


def test_missing_required_image_fields_are_reported(tmp_path):
    annotation_path, image_root = coco_fixture(tmp_path)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    del coco["images"][0]["id"]
    annotation_path.write_text(json.dumps(coco), encoding="utf-8")

    report = validate_doclaynet_split(annotation_path, image_root)

    assert "image record missing integer id or file_name" in report[
        "missing_or_invalid_records"
    ]


def test_missing_image_is_reported_as_invalid(tmp_path):
    annotations, image_root = coco_fixture(tmp_path, include_image=False)

    report = validate_doclaynet_split(annotations, image_root)

    assert report["matched_images"] == 0
    assert any("missing image page-1.png" in issue for issue in report["missing_or_invalid_records"])


def test_duplicate_ids_and_unmatched_annotations_are_detected(tmp_path):
    annotations, image_root = coco_fixture(tmp_path)
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    coco["annotations"].append(
        {**coco["annotations"][0], "id": 5, "image_id": 99}
    )
    annotations.write_text(json.dumps(coco), encoding="utf-8")

    report = validate_doclaynet_split(annotations, image_root)

    assert any("unknown image_id 99" in issue for issue in report["missing_or_invalid_records"])


def test_stratified_selection_is_repeatable_and_has_no_duplicate_pages():
    categories = (
        "financial_reports",
        "scientific_articles",
        "laws_and_regulations",
        "government_tenders",
        "manuals",
        "patents",
    )
    records = iter(
        [
            {"id": index, "doc_category": categories[index % len(categories)]}
            for index in range(60)
        ]
    )

    first = select_representative_images(records, 12, 42)
    second = select_representative_images(
        iter(
            [
                {"id": index, "doc_category": categories[index % len(categories)]}
                for index in range(60)
            ]
        ),
        12,
        42,
    )

    assert first == second
    assert len({record["id"] for record in first}) == 12
    assert {record["doc_category"] for record in first} == set(categories)


def test_duplicate_source_image_ids_are_rejected():
    records = [
        {"id": 1, "doc_category": category}
        for category in DOC_CATEGORIES
    ]

    with pytest.raises(SubsetDownloadError, match="duplicate image IDs"):
        select_representative_images(iter(records), len(DOC_CATEGORIES), 42)


def test_png_dimensions_reject_invalid_image(tmp_path):
    invalid = tmp_path / "broken.png"
    invalid.write_bytes(b"not png")

    with pytest.raises(ValueError, match="Not a complete PNG"):
        read_png_dimensions(invalid)


def test_image_extraction_closes_temp_file_before_atomic_replace(tmp_path):
    source_image = tmp_path / "source.png"
    write_png_header(source_image)
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(source_image, "PNG/source.png")

    destination = tmp_path / "output" / "source.png"
    with zipfile.ZipFile(archive_path) as archive:
        _extract_image_member(
            archive,
            archive.getinfo("PNG/source.png"),
            destination,
        )

    assert destination.read_bytes() == source_image.read_bytes()
    assert list(destination.parent.glob("*.tmp")) == []
