"""Local PDF and metadata persistence for the research paper collection."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from modules.pdf_processor import PDFExtractionError, ProcessedPaper, process_pdf

METADATA_VERSION = 1
VALID_STATUSES = {"processing", "ready", "failed", "missing"}
_STORE_LOCK = threading.RLock()


@dataclass(frozen=True)
class StoredPaper:
    paper_id: str
    original_filename: str
    stored_filename: str
    sha256: str
    page_count: int
    upload_date: str
    processing_status: str
    error: str = ""


@dataclass(frozen=True)
class StorageRestoreResult:
    papers: dict[str, ProcessedPaper]
    records: list[StoredPaper]
    warnings: list[str]


class LocalPaperStore:
    """Persist original PDFs and a recoverable JSON index inside the project."""

    def __init__(self, project_dir: str | Path | None = None) -> None:
        self.project_dir = (
            Path(project_dir).resolve()
            if project_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.upload_dir = self.project_dir / "data" / "uploads"
        self.metadata_path = self.project_dir / "data" / "papers.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _display_filename(filename: str) -> str:
        return Path(filename.replace("\\", "/")).name or "paper.pdf"

    @staticmethod
    def _recovered_display_name(stored_filename: str) -> str:
        stem = Path(stored_filename).stem
        recovered = re.sub(r"_[0-9a-f]{64}(?:_[0-9a-f]{8})?$", "", stem)
        return f"{recovered or 'paper'}.pdf"

    def _pdf_path(self, stored_filename: str) -> Path:
        if Path(stored_filename).name != stored_filename:
            raise ValueError("Stored PDF metadata contains an unsafe filename.")
        if Path(stored_filename).suffix.casefold() != ".pdf":
            raise ValueError("Stored PDF metadata points to a non-PDF file.")
        path = (self.upload_dir / stored_filename).resolve()
        if path.parent != self.upload_dir.resolve():
            raise ValueError("Stored PDF path is outside the local upload directory.")
        return path

    def _write_records(self, records: list[StoredPaper]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.metadata_path.with_name(
            f".{self.metadata_path.name}.{uuid4().hex}.tmp"
        )
        payload = {
            "version": METADATA_VERSION,
            "papers": [asdict(record) for record in records],
        }
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary_path, self.metadata_path)
        except OSError:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _load_records(
        self,
        *,
        recover_orphans: bool = False,
    ) -> tuple[list[StoredPaper], list[str]]:
        warnings: list[str] = []
        records: list[StoredPaper] = []
        known_hashes: set[str] = set()
        records_need_rewrite = False
        metadata_exists = self.metadata_path.exists()

        if metadata_exists:
            try:
                payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(payload, dict)
                    or payload.get("version") != METADATA_VERSION
                    or not isinstance(payload.get("papers"), list)
                ):
                    raise ValueError("Unsupported or malformed metadata structure.")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                warnings.append(
                    "Paper metadata could not be read; the local PDF directory is "
                    f"being scanned for recovery ({exc})."
                )
                payload = None

            if payload is not None:
                for entry in payload["papers"]:
                    try:
                        if not isinstance(entry, dict):
                            raise ValueError("record must be an object")
                        record = StoredPaper(
                            paper_id=str(entry["paper_id"]),
                            original_filename=str(entry["original_filename"]),
                            stored_filename=str(entry["stored_filename"]),
                            sha256=str(entry["sha256"]),
                            page_count=int(entry.get("page_count", 0)),
                            upload_date=str(entry["upload_date"]),
                            processing_status=str(entry["processing_status"]),
                            error=str(entry.get("error", "")),
                        )
                        if (
                            len(record.sha256) != 64
                            or record.paper_id != record.sha256
                            or record.processing_status not in VALID_STATUSES
                            or record.page_count < 0
                        ):
                            raise ValueError("record fields are invalid")
                        self._pdf_path(record.stored_filename)
                    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                        warnings.append(
                            f"An invalid paper metadata record was skipped ({exc})."
                        )
                        records_need_rewrite = True
                        continue
                    if record.sha256 in known_hashes:
                        warnings.append(
                            f"Duplicate metadata for {record.original_filename} was ignored."
                        )
                        records_need_rewrite = True
                        continue
                    records.append(record)
                    known_hashes.add(record.sha256)

        scanned = False
        should_scan = recover_orphans or not metadata_exists or payload is None
        pdf_paths = sorted(self.upload_dir.glob("*.pdf")) if should_scan else []
        for pdf_path in pdf_paths:
            if pdf_path.is_symlink():
                warnings.append(
                    f"Skipped symbolic-link PDF in local storage: {pdf_path.name}."
                )
                continue
            try:
                digest = sha256(pdf_path.read_bytes()).hexdigest()
                modified = datetime.fromtimestamp(
                    pdf_path.stat().st_mtime, timezone.utc
                ).isoformat()
            except OSError as exc:
                warnings.append(f"Could not inspect stored PDF {pdf_path.name}: {exc}")
                continue
            if digest in known_hashes:
                continue
            records.append(
                StoredPaper(
                    paper_id=digest,
                    original_filename=self._recovered_display_name(pdf_path.name),
                    stored_filename=pdf_path.name,
                    sha256=digest,
                    page_count=0,
                    upload_date=modified,
                    processing_status="processing",
                )
            )
            known_hashes.add(digest)
            scanned = True

        if scanned or records_need_rewrite:
            self._write_records(records)
        elif not metadata_exists and records:
            self._write_records(records)
        return records, warnings

    def list_records(self) -> list[StoredPaper]:
        """Return persisted metadata, recovering unindexed PDFs when possible."""
        with _STORE_LOCK:
            records, _ = self._load_records()
            return records

    @staticmethod
    def _safe_stem(filename: str) -> str:
        stem = Path(filename).stem
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
        return safe[:80] or "paper"

    def _write_pdf(self, path: Path, file_data: bytes) -> None:
        temporary_path = self.upload_dir / f".{uuid4().hex}.tmp"
        try:
            temporary_path.write_bytes(file_data)
            os.replace(temporary_path, path)
        except OSError:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def save_pdf(
        self,
        file_data: bytes,
        original_filename: str,
    ) -> tuple[StoredPaper, bool]:
        """Save a PDF once by SHA-256; return its record and duplicate flag."""
        digest = sha256(file_data).hexdigest()
        display_name = self._display_filename(original_filename)
        with _STORE_LOCK:
            records, _ = self._load_records()
            existing = next((item for item in records if item.sha256 == digest), None)
            if existing is not None:
                path = self._pdf_path(existing.stored_filename)
                try:
                    stored_digest = sha256(path.read_bytes()).hexdigest()
                except FileNotFoundError:
                    stored_digest = ""
                if stored_digest != digest:
                    stored_name = existing.stored_filename
                    if path.exists():
                        stored_name = (
                            f"{self._safe_stem(display_name)}_{digest}_"
                            f"{uuid4().hex[:8]}.pdf"
                        )
                    path = self._pdf_path(stored_name)
                    self._write_pdf(path, file_data)
                    existing = replace(
                        existing,
                        stored_filename=path.name,
                        processing_status="processing",
                        error="",
                    )
                    records = [
                        existing if record.paper_id == digest else record
                        for record in records
                    ]
                    self._write_records(records)
                return existing, True

            stored_name = f"{self._safe_stem(display_name)}_{digest}.pdf"
            path = self._pdf_path(stored_name)
            if path.exists():
                if sha256(path.read_bytes()).hexdigest() != digest:
                    stored_name = (
                        f"{self._safe_stem(display_name)}_{digest}_"
                        f"{uuid4().hex[:8]}.pdf"
                    )
                    path = self._pdf_path(stored_name)
            self._write_pdf(path, file_data)
            record = StoredPaper(
                paper_id=digest,
                original_filename=display_name,
                stored_filename=stored_name,
                sha256=digest,
                page_count=0,
                upload_date=datetime.now(timezone.utc).isoformat(),
                processing_status="processing",
            )
            records.append(record)
            self._write_records(records)
            return record, False

    def update_status(
        self,
        paper_id: str,
        status: str,
        *,
        page_count: int | None = None,
        error: str = "",
    ) -> StoredPaper:
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown paper processing status: {status}")
        with _STORE_LOCK:
            records, _ = self._load_records()
            updated: StoredPaper | None = None
            new_records: list[StoredPaper] = []
            for record in records:
                if record.paper_id == paper_id:
                    updated = replace(
                        record,
                        processing_status=status,
                        page_count=(
                            record.page_count if page_count is None else page_count
                        ),
                        error=error,
                    )
                    new_records.append(updated)
                else:
                    new_records.append(record)
            if updated is None:
                raise KeyError(f"No stored paper has ID {paper_id}.")
            self._write_records(new_records)
            return updated

    def restore(self) -> StorageRestoreResult:
        """Re-extract every available PDF and repair its processing status."""
        with _STORE_LOCK:
            records, warnings = self._load_records(recover_orphans=True)
            restored: dict[str, ProcessedPaper] = {}
            updated_records: list[StoredPaper] = []
            changed = False
            for record in records:
                try:
                    path = self._pdf_path(record.stored_filename)
                except (OSError, ValueError) as exc:
                    message = f"Unsafe or unavailable PDF path for {record.original_filename}: {exc}"
                    warnings.append(message)
                    updated_records.append(
                        replace(record, processing_status="failed", error=message)
                    )
                    changed |= record.processing_status != "failed" or record.error != message
                    continue
                try:
                    file_data = path.read_bytes()
                except FileNotFoundError:
                    message = f"Stored PDF is missing: {record.original_filename}."
                    warnings.append(message)
                    updated_records.append(
                        replace(
                            record,
                            processing_status="missing",
                            error=message,
                        )
                    )
                    changed |= record.processing_status != "missing" or record.error != message
                    continue
                except OSError as exc:
                    message = f"Could not read {record.original_filename}: {exc}"
                    warnings.append(message)
                    updated_records.append(
                        replace(record, processing_status="failed", error=message)
                    )
                    changed |= record.processing_status != "failed" or record.error != message
                    continue

                if sha256(file_data).hexdigest() != record.sha256:
                    message = (
                        f"Stored PDF checksum does not match metadata: "
                        f"{record.original_filename}."
                    )
                    warnings.append(message)
                    updated_records.append(
                        replace(record, processing_status="failed", error=message)
                    )
                    changed |= record.processing_status != "failed" or record.error != message
                    continue

                try:
                    paper = process_pdf(file_data, record.original_filename)
                except PDFExtractionError as exc:
                    message = str(exc)
                    warnings.append(
                        f"Could not process {record.original_filename}: {message}"
                    )
                    updated_records.append(
                        replace(record, processing_status="failed", error=message)
                    )
                    changed |= record.processing_status != "failed" or record.error != message
                    continue
                restored[record.paper_id] = paper
                updated = replace(
                    record,
                    page_count=paper.page_count,
                    processing_status="ready",
                    error="",
                )
                updated_records.append(updated)
                changed |= updated != record

            if changed:
                self._write_records(updated_records)
            return StorageRestoreResult(restored, updated_records, warnings)

    def remove_papers(self, paper_ids: list[str] | set[str]) -> tuple[list[str], list[str]]:
        """Remove only selected records and their corresponding PDF files."""
        selected_ids = set(paper_ids)
        if not selected_ids:
            return [], []
        with _STORE_LOCK:
            records, _ = self._load_records()
            removed: list[str] = []
            errors: list[str] = []
            retained: list[StoredPaper] = []
            for record in records:
                if record.paper_id not in selected_ids:
                    retained.append(record)
                    continue
                try:
                    path = self._pdf_path(record.stored_filename)
                    path.unlink(missing_ok=True)
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"Could not remove {record.original_filename}: {exc}"
                    )
                    retained.append(record)
                    continue
                removed.append(record.paper_id)
            if removed:
                self._write_records(retained)
            return removed, errors
