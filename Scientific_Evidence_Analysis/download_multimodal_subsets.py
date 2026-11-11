"""Download small official DocLayNet samples without fetching its full ZIP.

PubTables-1M exposes only full tar.gz archives on its official Hugging Face
repository. This script deliberately does not download any of those archives.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import struct
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from modules.multimodal_data import read_png_dimensions, validate_doclaynet_split

PROJECT_ROOT = Path(__file__).resolve().parent
DOCLAYNET_DIR = PROJECT_ROOT / "datasets" / "benchmarks" / "doclaynet" / "data"
PUBTABLES_DIR = PROJECT_ROOT / "datasets" / "benchmarks" / "pubtables" / "data"
DOCLAYNET_CORE_URL = (
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
    "dax-doclaynet/1.0.0/DocLayNet_core.zip"
)
DOCLAYNET_ORIGINAL_BYTES = 30_012_083_650
DOCLAYNET_SPLITS = ("train", "val", "test")
DOC_CATEGORIES = (
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
)
CHUNK_SIZE = 1024 * 1024


class SubsetDownloadError(RuntimeError):
    """Raised when an official subset cannot be downloaded or validated."""


class HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded-cache reader over HTTP Range responses."""

    def __init__(
        self,
        url: str,
        size: int,
        *,
        block_size: int = CHUNK_SIZE,
        max_cached_blocks: int = 4,
        retries: int = 3,
    ) -> None:
        super().__init__()
        self.url = url
        self.size = size
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.retries = retries
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.range_requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unknown seek mode: {whence}")
        if position < 0:
            raise ValueError("Negative seek position.")
        self.position = position
        return self.position

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]

        start = index * self.block_size
        end = min(self.size, start + self.block_size) - 1
        if start > end:
            return b""
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "ScientificEvidenceAnalysis-DocLayNetSubset/1.0",
            },
        )
        error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise SubsetDownloadError(
                            "Official DocLayNet storage did not honor a byte-range "
                            f"request (HTTP {response.status}); refusing full archive download."
                        )
                    expected_range = f"bytes {start}-{end}/"
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(expected_range):
                        raise SubsetDownloadError(
                            "Unexpected Content-Range from DocLayNet storage: "
                            f"{content_range!r}."
                        )
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError(
                        f"Short range read: expected {end - start + 1} bytes, "
                        f"received {len(data)}."
                    )
                self.cache[index] = data
                self.cache.move_to_end(index)
                while len(self.cache) > self.max_cached_blocks:
                    self.cache.popitem(last=False)
                self.range_requests += 1
                if self.range_requests % 25 == 0:
                    print(
                        f"Retrieved {self.range_requests} archive ranges.",
                        flush=True,
                    )
                return data
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                error = exc
                if attempt < self.retries - 1:
                    time.sleep(attempt + 1)
        if isinstance(error, SubsetDownloadError):
            raise error
        raise SubsetDownloadError(
            f"Could not retrieve DocLayNet archive bytes {start}-{end}: {error}"
        ) from error

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = max(0, self.size - self.position)
        size = min(size, max(0, self.size - self.position))
        end_position = self.position + size
        output = bytearray()
        while self.position < end_position:
            index = self.position // self.block_size
            data = self._block(index)
            offset = self.position - index * self.block_size
            count = min(end_position - self.position, len(data) - offset)
            if count <= 0:
                break
            output.extend(data[offset:offset + count])
            self.position += count
        return bytes(output)

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def _official_archive_info() -> tuple[int, str]:
    request = urllib.request.Request(
        DOCLAYNET_CORE_URL,
        method="HEAD",
        headers={"User-Agent": "ScientificEvidenceAnalysis-DocLayNetSubset/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            size = int(response.headers["Content-Length"])
            etag = response.headers.get("ETag", "")
            if not response.headers.get("Accept-Ranges", "").casefold().startswith("bytes"):
                raise SubsetDownloadError(
                    "Official DocLayNet storage does not advertise byte-range "
                    "support. The 30 GB archive will not be downloaded."
                )
    except (OSError, TimeoutError, KeyError, ValueError, urllib.error.URLError) as exc:
        if isinstance(exc, SubsetDownloadError):
            raise
        raise SubsetDownloadError(
            f"Could not inspect official DocLayNet archive headers: {exc}"
        ) from exc
    if size != DOCLAYNET_ORIGINAL_BYTES:
        print(
            "Notice: official archive size differs from the checked repository "
            f"value ({size:,} bytes vs. {DOCLAYNET_ORIGINAL_BYTES:,})."
        )
    return size, etag


class JSONStreamReader:
    """Incremental parser for a top-level JSON object containing large arrays."""

    def __init__(self, stream: Any, chunk_size: int = CHUNK_SIZE) -> None:
        self.stream = stream
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.position = 0

    def _fill(self) -> bool:
        data = self.stream.read(self.chunk_size)
        if not data:
            return False
        self.buffer += data.decode("utf-8")
        return True

    def _skip_space(self) -> None:
        while True:
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position].isspace()
            ):
                self.position += 1
            if self.position < len(self.buffer):
                return
            if not self._fill():
                return

    def _decode_value(self) -> Any:
        while True:
            try:
                value, next_position = self.decoder.raw_decode(
                    self.buffer,
                    self.position,
                )
                self.position = next_position
                return value
            except json.JSONDecodeError:
                if not self._fill():
                    raise SubsetDownloadError(
                        "Malformed or truncated JSON in official COCO annotations."
                    )

    def _compact(self) -> None:
        if self.position >= self.chunk_size:
            self.buffer = self.buffer[self.position:]
            self.position = 0

    def _array_items(self) -> Iterator[Any]:
        while True:
            self._skip_space()
            if self.position >= len(self.buffer):
                raise SubsetDownloadError("COCO JSON array ended unexpectedly.")
            if self.buffer[self.position] == "]":
                self.position += 1
                self._compact()
                return
            if self.buffer[self.position] == ",":
                self.position += 1
                continue
            value = self._decode_value()
            yield value
            self._compact()

    def items(self) -> Iterator[tuple[str, Any]]:
        """Yield object fields; array values are lazy iterators consumed in order."""
        self._skip_space()
        if self.position >= len(self.buffer) or self.buffer[self.position] != "{":
            raise SubsetDownloadError("Expected a top-level COCO JSON object.")
        self.position += 1
        while True:
            self._skip_space()
            if self.position >= len(self.buffer):
                raise SubsetDownloadError("COCO JSON object ended unexpectedly.")
            if self.buffer[self.position] == "}":
                return
            if self.buffer[self.position] == ",":
                self.position += 1
                continue
            key = self._decode_value()
            if not isinstance(key, str):
                raise SubsetDownloadError("COCO JSON object key is not a string.")
            self._skip_space()
            if self.position >= len(self.buffer) or self.buffer[self.position] != ":":
                raise SubsetDownloadError(f"COCO JSON key {key!r} has no colon.")
            self.position += 1
            self._skip_space()
            if self.position >= len(self.buffer):
                raise SubsetDownloadError(f"COCO JSON key {key!r} has no value.")
            if self.buffer[self.position] == "[":
                self.position += 1
                value: Any = self._array_items()
            else:
                value = self._decode_value()
            yield key, value
            if not isinstance(value, Iterator):
                self._compact()


def _drain(value: Any) -> None:
    if isinstance(value, Iterator):
        for _ in value:
            pass


def iter_json_array(
    stream: Any,
    key: str,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[Any]:
    """Incrementally parse a named top-level JSON array from a COCO stream."""
    parser = JSONStreamReader(stream, chunk_size)
    for field, value in parser.items():
        if field == key:
            if not isinstance(value, Iterator):
                raise SubsetDownloadError(f"COCO JSON field {key!r} is not an array.")
            yield from value
            return
        _drain(value)
    raise SubsetDownloadError(f"COCO JSON array field {key!r} was not found.")


def _reservoirs_to_sample(
    reservoirs: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    for category, quota in quotas.items():
        candidates = reservoirs.get(category, [])
        amount = min(quota, len(candidates))
        selected = rng.sample(candidates, amount)
        chosen.extend(selected)
        chosen_ids.update(int(record["id"]) for record in selected)
    if len(chosen) < target:
        remaining = [
            record
            for records in reservoirs.values()
            for record in records
            if int(record["id"]) not in chosen_ids
        ]
        rng.shuffle(remaining)
        chosen.extend(remaining[:target - len(chosen)])
    if len(chosen) != target:
        raise SubsetDownloadError(
            f"Could only select {len(chosen)} of {target} requested pages."
        )
    return chosen


def select_representative_images(
    records: Iterator[dict[str, Any]],
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministic reservoir sample balanced across DocLayNet document types."""
    if target < 1:
        raise ValueError("The DocLayNet target must be positive.")
    base, remainder = divmod(target, len(DOC_CATEGORIES))
    category_order = list(DOC_CATEGORIES)
    random.Random(seed).shuffle(category_order)
    quotas = {
        category: base + (index < remainder)
        for index, category in enumerate(category_order)
    }
    capacity = max(quotas.values(), default=0)
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    rng = random.Random(seed)
    for record in records:
        category = record.get("doc_category")
        if category not in DOC_CATEGORIES:
            continue
        counts[category] += 1
        reservoir = reservoirs[category]
        if len(reservoir) < capacity:
            reservoir.append(record)
            continue
        replacement = rng.randrange(counts[category])
        if replacement < capacity:
            reservoir[replacement] = record
    sample = _reservoirs_to_sample(reservoirs, quotas, target, rng)
    if len({int(record["id"]) for record in sample}) != target:
        raise SubsetDownloadError("DocLayNet sample contains duplicate image IDs.")
    return sample


def _split_quotas(total: int) -> dict[str, int]:
    if total < len(DOCLAYNET_SPLITS):
        raise ValueError("At least three pages are needed to include each split.")
    base, remainder = divmod(total, len(DOCLAYNET_SPLITS))
    quotas = {split: base for split in DOCLAYNET_SPLITS}
    for split in DOCLAYNET_SPLITS[:remainder]:
        quotas[split] += 1
    return quotas


def _safe_member_path(filename: str) -> str:
    path = Path(filename)
    if path.name != filename or filename in {"", ".", ".."}:
        raise SubsetDownloadError(f"Unsafe image name in official annotations: {filename!r}")
    return f"PNG/{filename}"


def _image_matches_zip_info(path: Path, info: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_SIZE)
            if not chunk:
                break
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF == info.CRC


def _extract_image_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> None:
    if _image_matches_zip_info(destination, info):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.stem}-",
        suffix=".png.tmp",
        dir=destination.parent,
        delete=False,
    ) as output:
        temporary_path = Path(output.name)
    try:
        with archive.open(info, "r") as source, temporary_path.open("wb") as output:
            shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        if not _image_matches_zip_info(temporary_path, info):
            raise SubsetDownloadError(
                f"Image CRC or size validation failed for {info.filename}."
            )
        read_png_dimensions(temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _subset_is_complete(total: int, etag: str) -> bool:
    manifest_path = DOCLAYNET_DIR / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_etag") != etag
            or manifest.get("requested_page_count") != total
        ):
            return False
        for split, expected in manifest["splits"].items():
            annotation_path = DOCLAYNET_DIR / "annotations" / f"{split}.json"
            image_root = DOCLAYNET_DIR / "images" / split
            if not annotation_path.is_file():
                return False
            report = validate_doclaynet_split(annotation_path, image_root)
            if report["record_count"] != expected["record_count"]:
                return False
            if report["missing_or_invalid_records"]:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def download_doclaynet_subset(total: int = 150, seed: int = 2026) -> dict[str, Any]:
    """Fetch balanced DocLayNet annotations and image members via HTTP ranges."""
    archive_size, etag = _official_archive_info()
    if _subset_is_complete(total, etag):
        print(f"Reusing validated DocLayNet {total}-page subset.")
        return json.loads(
            (DOCLAYNET_DIR / "manifest.json").read_text(encoding="utf-8")
        )

    quotas = _split_quotas(total)
    range_reader = HTTPRangeReader(DOCLAYNET_CORE_URL, archive_size)
    split_results: dict[str, Any] = {}
    with zipfile.ZipFile(range_reader, "r") as archive:
        for split in DOCLAYNET_SPLITS:
            coco_member = f"COCO/{'val' if split == 'val' else split}.json"
            try:
                coco_info = archive.getinfo(coco_member)
            except KeyError as exc:
                raise SubsetDownloadError(
                    f"Official DocLayNet archive is missing {coco_member}."
                ) from exc
            images: list[dict[str, Any]] | None = None
            info: Any = {}
            licenses: list[Any] = []
            categories: list[Any] = []
            annotation_records: list[dict[str, Any]] = []
            with archive.open(coco_info, "r") as stream:
                parser = JSONStreamReader(stream)
                for key, value in parser.items():
                    if key == "info":
                        info = value
                    elif key == "licenses":
                        licenses = list(value) if isinstance(value, Iterator) else []
                    elif key == "images":
                        images = select_representative_images(
                            value,
                            quotas[split],
                            seed + sum(split.encode("utf-8")),
                        )
                    elif key == "annotations":
                        if images is None:
                            raise SubsetDownloadError(
                                f"{coco_member} lists annotations before images."
                            )
                        selected_ids = {int(record["id"]) for record in images}
                        annotation_records = [
                            annotation
                            for annotation in value
                            if annotation.get("image_id") in selected_ids
                        ]
                    elif key == "categories":
                        categories = list(value) if isinstance(value, Iterator) else []
                    else:
                        _drain(value)
            if images is None:
                raise SubsetDownloadError(f"{coco_member} contains no images array.")

            image_root = DOCLAYNET_DIR / "images" / split
            for index, image in enumerate(images, start=1):
                filename = image.get("file_name")
                if not isinstance(filename, str):
                    raise SubsetDownloadError(
                        f"Selected DocLayNet image record has no filename: {image!r}"
                    )
                member_name = _safe_member_path(filename)
                try:
                    member_info = archive.getinfo(member_name)
                except KeyError as exc:
                    raise SubsetDownloadError(
                        f"Image {member_name} is missing from the official ZIP."
                    ) from exc
                _extract_image_member(
                    archive,
                    member_info,
                    image_root / filename,
                )
                if index % 25 == 0:
                    print(f"{split}: prepared {index}/{quotas[split]} pages.")

            coco_subset = {
                "info": info,
                "licenses": licenses,
                "images": images,
                "annotations": annotation_records,
                "categories": categories,
            }
            annotation_path = DOCLAYNET_DIR / "annotations" / f"{split}.json"
            _atomic_json(annotation_path, coco_subset)
            report = validate_doclaynet_split(annotation_path, image_root)
            if report["missing_or_invalid_records"]:
                raise SubsetDownloadError(
                    f"DocLayNet {split} subset failed validation: "
                    f"{report['missing_or_invalid_records'][:5]}"
                )
            split_results[split] = {
                "record_count": report["record_count"],
                "annotation_count": report["annotation_count"],
                "category_distribution": {
                    category: sum(
                        item.get("doc_category") == category for item in images
                    )
                    for category in DOC_CATEGORIES
                },
                "annotation_types": report["annotation_types"],
                "source_split": "val" if split == "val" else split,
            }
            del annotation_records, categories, licenses, info, images
            print(
                f"DocLayNet {split}: {report['record_count']} pages, "
                f"{report['annotation_count']} layout annotations."
            )

    manifest = {
        "dataset": "DocLayNet",
        "source": "official DocLayNet core ZIP via HTTP byte ranges",
        "source_url": DOCLAYNET_CORE_URL,
        "source_etag": etag,
        "original_archive_bytes": archive_size,
        "download_date_utc": datetime.now(timezone.utc).isoformat(),
        "requested_page_count": total,
        "sampling_seed": seed,
        "sampling": "deterministic reservoir sampling balanced across document categories",
        "splits": split_results,
    }
    _atomic_json(DOCLAYNET_DIR / "manifest.json", manifest)
    return manifest


def record_pubtables_limitation() -> dict[str, Any]:
    """Record why PubTables-1M is not downloaded as an oversized full archive."""
    PUBTABLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = PUBTABLES_DIR / "subset_status.json"
    if status_path.is_file():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    status = {
        "dataset": "PubTables-1M",
        "source_dataset": "https://huggingface.co/datasets/bsmock/pubtables-1m",
        "official_repository": "https://github.com/microsoft/table-transformer",
        "download_date_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_records": 0,
        "status": "not_downloaded",
        "reason": (
            "The official Hugging Face dataset card states that the dataset is "
            "provided only as 18 tar.gz archives and that Hugging Face Datasets "
            "integration is not supported. The viewer exposes no splits or "
            "Parquet files. The annotation and matching image files are only "
            "available inside large split archives, with no official per-table "
            "sample endpoint. Downloading those full archives would violate the "
            "requested resource limits, so no PubTables data archive was fetched."
        ),
        "license": "CDLA-Permissive-2.0",
    }
    _atomic_json(status_path, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doclaynet-pages",
        type=int,
        default=150,
        help="Total balanced DocLayNet pages to prepare (default: 150).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for deterministic category-stratified sampling.",
    )
    args = parser.parse_args()
    try:
        manifest = download_doclaynet_subset(args.doclaynet_pages, args.seed)
        pubtables = record_pubtables_limitation()
    except (OSError, ValueError, SubsetDownloadError, zipfile.BadZipFile) as exc:
        print(f"Multimodal subset preparation failed: {exc}")
        return 1

    print(f"\nDocLayNet subset: {DOCLAYNET_DIR}")
    print(f"Original core archive size: {manifest['original_archive_bytes']:,} bytes")
    print(
        "Sample records: "
        + ", ".join(
            f"{split}={data['record_count']}"
            for split, data in manifest["splits"].items()
        )
    )
    print(
        "PubTables-1M records downloaded: "
        f"{pubtables['downloaded_records']} (see subset_status.json for the "
        "official-format limitation)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
