"""Download and locally prepare the official QASPER dataset splits."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk, load_dataset
from datasets.exceptions import DatasetBuildError
import pyarrow

DATASET_ID = "allenai/qasper"
VIEWER_PARQUET_API = "https://datasets-server.huggingface.co/parquet"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "datasets" / "benchmarks" / "qasper" / "data"
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"
MANIFEST_PATH = DATA_DIR / "download_manifest.json"


class QasperDownloadError(RuntimeError):
    """Raised when the official dataset cannot be downloaded or prepared."""


def get_official_parquet_files(timeout: int = 60) -> dict[str, dict[str, Any]]:
    """Get official converted Parquet URLs from the Hugging Face dataset viewer."""
    url = f"{VIEWER_PARQUET_API}?dataset={DATASET_ID.replace('/', '%2F')}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ScientificEvidenceAnalysis-QASPER/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        OSError,
        TimeoutError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ) as exc:
        raise QasperDownloadError(
            "Could not retrieve the official QASPER Parquet listing from the "
            f"Hugging Face dataset viewer: {exc}"
        ) from exc

    files: dict[str, dict[str, Any]] = {}
    for entry in payload.get("parquet_files", []):
        if entry.get("dataset") != DATASET_ID or entry.get("config") != "qasper":
            continue
        split = entry.get("split")
        url_value = entry.get("url")
        size = entry.get("size")
        if not isinstance(split, str) or not isinstance(url_value, str):
            continue
        if not isinstance(size, int) or size <= 0:
            continue
        if split in files:
            raise QasperDownloadError(
                f"The official viewer returned multiple Parquet files for {split}; "
                "this script expects one consolidated file per split."
            )
        files[split] = {
            "url": url_value,
            "size": size,
            "filename": entry.get("filename") or "0000.parquet",
        }

    if not files:
        raise QasperDownloadError(
            "The Hugging Face viewer did not provide downloadable QASPER Parquet "
            "files. The repository currently exposes a legacy loader and dummy "
            "archive, which are not the full dataset."
        )
    return files


def _is_valid_parquet(path: Path, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    try:
        import pyarrow.parquet as parquet

        parquet.read_metadata(path)
    except (ImportError, OSError, ValueError, pyarrow.ArrowException):
        return False
    return True


def download_raw_file(
    split: str,
    file_info: dict[str, Any],
    timeout: int = 120,
) -> Path:
    """Download a single official raw Parquet file safely and idempotently."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / f"{split}-0000.parquet"
    expected_size = int(file_info["size"])
    if _is_valid_parquet(destination, expected_size):
        print(f"Reusing existing {split} Parquet ({expected_size:,} bytes).")
        return destination

    request = urllib.request.Request(
        file_info["url"],
        headers={"User-Agent": "ScientificEvidenceAnalysis-QASPER/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        temporary_path: Path | None = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{split}-",
                    suffix=".parquet.tmp",
                    dir=RAW_DIR,
                    delete=False,
                ) as output:
                    temporary_path = Path(output.name)
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            actual_size = temporary_path.stat().st_size
            if actual_size != expected_size:
                raise QasperDownloadError(
                    f"Downloaded {split} data has {actual_size:,} bytes; the "
                    f"official listing specified {expected_size:,} bytes."
                )
            if not _is_valid_parquet(temporary_path, expected_size):
                raise QasperDownloadError(
                    f"The downloaded {split} file is not a readable Parquet file."
                )
            os.replace(temporary_path, destination)
            print(
                f"Downloaded {split}: {destination} ({expected_size:,} bytes)."
            )
            return destination
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            QasperDownloadError,
        ) as exc:
            last_error = exc
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(attempt + 1)

    raise QasperDownloadError(
        f"Could not download a complete, valid official {split} split after "
        f"three attempts: {last_error}"
    ) from last_error


def load_raw_dataset(raw_files: dict[str, Path]) -> DatasetDict:
    """Load downloaded official Parquet data with Hugging Face's Parquet builder."""
    try:
        dataset = load_dataset(
            "parquet",
            data_files={split: str(path) for split, path in raw_files.items()},
        )
    except (
        DatasetBuildError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pyarrow.ArrowException,
    ) as exc:
        raise QasperDownloadError(
            "Hugging Face datasets could not load the downloaded QASPER Parquet "
            f"files: {exc}"
        ) from exc
    if not isinstance(dataset, DatasetDict) or not dataset:
        raise QasperDownloadError(
            "The Parquet loader returned no named QASPER dataset splits."
        )
    return dataset


def _load_prepared_if_valid(
    file_listing: dict[str, dict[str, Any]],
) -> DatasetDict | None:
    if not MANIFEST_PATH.is_file() or not PREPARED_DIR.is_dir():
        return None
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if (
            manifest.get("dataset") != DATASET_ID
            or manifest.get("parquet_files") != file_listing
        ):
            return None
        dataset = load_from_disk(str(PREPARED_DIR))
        if not isinstance(dataset, DatasetDict):
            return None
        if set(dataset) != set(manifest.get("record_counts", {})):
            return None
        if {
            split: len(records) for split, records in dataset.items()
        } != manifest["record_counts"]:
            return None
        return dataset
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        pyarrow.ArrowException,
    ):
        return None


def save_prepared_dataset(
    dataset: DatasetDict,
    prepared_dir: str | Path = PREPARED_DIR,
) -> None:
    prepared_path = Path(prepared_dir)
    data_dir = prepared_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".qasper-prepared-", dir=data_dir)
    )
    backup_dir = data_dir / f".{prepared_path.name}-previous"
    try:
        dataset.save_to_disk(str(temporary_dir))
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if prepared_path.exists():
            os.replace(prepared_path, backup_dir)
        try:
            os.replace(temporary_dir, prepared_path)
        except OSError:
            if backup_dir.exists() and not prepared_path.exists():
                os.replace(backup_dir, prepared_path)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def load_prepared_dataset(prepared_dir: str | Path = PREPARED_DIR) -> DatasetDict:
    """Reload a local dataset saved by :func:`save_prepared_dataset`."""
    dataset = load_from_disk(str(prepared_dir))
    if not isinstance(dataset, DatasetDict):
        raise QasperDownloadError(
            f"Expected a saved DatasetDict at {prepared_dir}."
        )
    return dataset


def prepare_qasper_dataset() -> DatasetDict:
    """Download raw splits and create/reuse a reloadable local DatasetDict."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_listing = get_official_parquet_files()
    existing_dataset = _load_prepared_if_valid(file_listing)
    if existing_dataset is not None and all(
        _is_valid_parquet(
            RAW_DIR / f"{split}-0000.parquet",
            int(file_info["size"]),
        )
        for split, file_info in file_listing.items()
    ):
        print("A complete local QASPER download was found; reusing it.")
        return existing_dataset

    raw_files = {
        split: download_raw_file(split, file_info)
        for split, file_info in file_listing.items()
    }
    dataset = load_raw_dataset(raw_files)
    save_prepared_dataset(dataset)
    record_counts = {split: len(records) for split, records in dataset.items()}
    manifest = {
        "dataset": DATASET_ID,
        "source": "Hugging Face datasets-server Parquet export",
        "data_directory": str(DATA_DIR),
        "parquet_files": file_listing,
        "record_counts": record_counts,
    }
    temporary_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, MANIFEST_PATH)
    return dataset


def main() -> int:
    try:
        dataset = prepare_qasper_dataset()
    except QasperDownloadError as exc:
        print(f"QASPER download failed: {exc}")
        return 1
    except (OSError, ValueError, RuntimeError, pyarrow.ArrowException) as exc:
        print(f"QASPER local preparation failed: {exc}")
        return 1

    print(f"\nQASPER saved under: {DATA_DIR}")
    print(f"Reloadable Hugging Face DatasetDict: {PREPARED_DIR}")
    for split, records in dataset.items():
        print(f"{split}: {len(records):,} paper records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
