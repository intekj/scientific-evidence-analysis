"""Inspect local DocLayNet and PubTables-1M benchmark subsets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.multimodal_data import validate_doclaynet_split

PROJECT_ROOT = Path(__file__).resolve().parent
DOCLAYNET_DIR = PROJECT_ROOT / "datasets" / "benchmarks" / "doclaynet" / "data"
PUBTABLES_DIR = PROJECT_ROOT / "datasets" / "benchmarks" / "pubtables" / "data"
SAMPLE_LIMIT = 3


def directory_size(path: str | Path) -> int:
    directory = Path(path)
    if not directory.exists():
        return 0
    return sum(
        file_path.stat().st_size
        for file_path in directory.rglob("*")
        if file_path.is_file()
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def inspect_doclaynet() -> dict[str, Any]:
    manifest_path = DOCLAYNET_DIR / "manifest.json"
    if not manifest_path.is_file():
        print("Dataset: DocLayNet")
        print("Downloaded sample count: 0 (subset manifest not found)")
        print(f"Location: {DOCLAYNET_DIR}")
        return {"dataset": "DocLayNet", "record_count": 0, "missing": True}

    manifest = _read_json(manifest_path)
    total_records = 0
    total_annotations = 0
    invalid_count = 0
    all_fields: set[str] = set()
    all_annotation_types: set[str] = set()
    all_categories: set[str] = set()
    samples: list[dict[str, Any]] = []
    split_results: dict[str, Any] = {}

    print("\nDataset: DocLayNet")
    print(f"Original core size: {manifest.get('original_archive_bytes', 0):,} bytes")
    print(f"Requested sample count: {manifest.get('requested_page_count', 0):,}")
    print(f"Source: {manifest.get('source_url')}")
    for split, expected in manifest.get("splits", {}).items():
        annotations_path = DOCLAYNET_DIR / "annotations" / f"{split}.json"
        image_root = DOCLAYNET_DIR / "images" / split
        try:
            report = validate_doclaynet_split(annotations_path, image_root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  {split}: INVALID ({exc})")
            invalid_count += 1
            continue
        invalid_count += len(report["missing_or_invalid_records"])
        total_records += report["record_count"]
        total_annotations += report["annotation_count"]
        all_fields.update(report["fields"])
        all_annotation_types.update(report["annotation_types"])
        all_categories.update(report["category_names"])
        split_results[split] = {
            "record_count": report["record_count"],
            "annotation_count": report["annotation_count"],
            "missing_or_invalid_count": len(report["missing_or_invalid_records"]),
            "image_dimensions": report["image_dimensions"],
        }
        print(
            f"  {split}: {report['record_count']} pages, "
            f"{report['annotation_count']} objects, "
            f"{len(report['missing_or_invalid_records'])} missing/invalid"
        )
        dimensions = sorted(
            {tuple(size) for size in report["image_dimensions"].values()}
        )
        print(f"    image dimensions: {dimensions or 'none'}")
        print(
            "    categories: "
            + ", ".join(report["category_names"])
        )
        if report["missing_or_invalid_records"]:
            for issue in report["missing_or_invalid_records"][:SAMPLE_LIMIT]:
                print(f"    INVALID: {issue}")
        annotation_path_value = annotations_path
        with annotation_path_value.open("r", encoding="utf-8") as stream:
            coco = json.load(stream)
        image_by_id = {image["id"]: image for image in coco.get("images", [])}
        for annotation in coco.get("annotations", [])[:SAMPLE_LIMIT]:
            if len(samples) < SAMPLE_LIMIT:
                samples.append(
                    {
                        "split": split,
                        "image": image_by_id.get(annotation.get("image_id")),
                        "annotation": annotation,
                    }
                )

    print(f"Validated downloaded sample count: {total_records:,}")
    print(f"Available record fields (images/annotations): {sorted(all_fields)}")
    print(f"Available annotation types: {sorted(all_annotation_types)}")
    print(f"Available class labels: {sorted(all_categories)}")
    print(f"Example annotations (up to {SAMPLE_LIMIT}):")
    for sample in samples:
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    print(f"Storage size: {directory_size(DOCLAYNET_DIR):,} bytes")
    print(f"Download date (UTC): {manifest.get('download_date_utc', 'not recorded')}")
    print(f"Missing/invalid records: {invalid_count}")
    return {
        "dataset": "DocLayNet",
        "record_count": total_records,
        "annotation_count": total_annotations,
        "missing_or_invalid_count": invalid_count,
        "fields": sorted(all_fields),
        "annotation_types": sorted(all_annotation_types),
        "categories": sorted(all_categories),
        "splits": split_results,
        "storage_bytes": directory_size(DOCLAYNET_DIR),
        "samples": samples,
    }


def _inspect_pubtables_records(annotation_path: Path) -> dict[str, Any]:
    """Inspect an optional locally supplied PubTables COCO/VOC/JSON sample."""
    if annotation_path.suffix.casefold() == ".json":
        value = _read_json(annotation_path)
        records = value.get("tables", value.get("annotations", []))
        if not isinstance(records, list):
            raise ValueError("PubTables JSON must contain a table list.")
        return {
            "record_count": len(records),
            "fields": sorted(
                {
                    field
                    for record in records[:SAMPLE_LIMIT]
                    if isinstance(record, dict)
                    for field in record
                }
            ),
            "samples": records[:SAMPLE_LIMIT],
        }
    return {
        "record_count": 0,
        "fields": [],
        "samples": [],
    }


def inspect_pubtables() -> dict[str, Any]:
    status_path = PUBTABLES_DIR / "subset_status.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    annotation_files = sorted(PUBTABLES_DIR.rglob("*.json"))
    annotation_files = [
        path for path in annotation_files if path.name != status_path.name
    ]
    record_count = 0
    fields: set[str] = set()
    samples: list[Any] = []
    invalid_count = 0
    for annotation_path in annotation_files:
        try:
            result = _inspect_pubtables_records(annotation_path)
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_count += 1
            continue
        record_count += result["record_count"]
        fields.update(result["fields"])
        samples.extend(result["samples"][:SAMPLE_LIMIT - len(samples)])
        if len(samples) >= SAMPLE_LIMIT:
            break

    print("\nDataset: PubTables-1M")
    print(f"Downloaded sample count: {record_count}")
    print(
        "Source: "
        + str(
            status.get(
                "source_dataset",
                "https://huggingface.co/datasets/bsmock/pubtables-1m",
            )
        )
    )
    print(f"Split distribution: {status.get('split_distribution', 'not available')}")
    print(f"Available fields: {sorted(fields) or 'no local annotations'}")
    print(
        "Available annotation types: "
        + (
            "table JSON/XML structures"
            if record_count
            else "none locally (no subset downloaded)"
        )
    )
    print(f"Storage size: {directory_size(PUBTABLES_DIR):,} bytes")
    print(f"Missing/invalid records: {invalid_count}")
    if status.get("status") == "not_downloaded":
        print("Subset limitation: " + status.get("reason", "not documented"))
    if samples:
        print(f"Example annotations (up to {SAMPLE_LIMIT}):")
        for sample in samples:
            print(json.dumps(sample, ensure_ascii=False, indent=2)[:2500])
    return {
        "dataset": "PubTables-1M",
        "record_count": record_count,
        "missing_or_invalid_count": invalid_count,
        "fields": sorted(fields),
        "storage_bytes": directory_size(PUBTABLES_DIR),
        "samples": samples,
        "status": status.get("status", "unknown"),
    }


def main() -> int:
    results = [inspect_doclaynet(), inspect_pubtables()]
    if results[0]["missing_or_invalid_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
