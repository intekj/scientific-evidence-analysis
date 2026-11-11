
"""Validate and inspect the locally downloaded SciFact dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "benchmarks"
    / "scifact"
    / "data"
)

FILES = {
    "corpus": "corpus.jsonl",
    "train": "claims_train.jsonl",
    "dev": "claims_dev.jsonl",
    "test": "claims_test.jsonl",
}

VALID_LABELS = {
    "SUPPORT",
    "CONTRADICT",
    "NOT_ENOUGH_INFO",
}


class SciFactValidationError(RuntimeError):
    """Raised when SciFact validation fails."""


# ==================================================
# 2. LOAD JSONL FILES
# ==================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records and validate their JSON structure."""

    if not path.is_file():
        raise SciFactValidationError(
            f"File not found: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                raise SciFactValidationError(
                    f"Empty line in {path.name}, line {line_number}."
                )

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise SciFactValidationError(
                    f"Invalid JSON in {path.name}, "
                    f"line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise SciFactValidationError(
                    f"Expected JSON object in {path.name}, "
                    f"line {line_number}."
                )

            records.append(record)

    return records


# ==================================================
# 3. VALIDATE CORPUS
# ==================================================

def validate_corpus(
    corpus: list[dict[str, Any]],
) -> tuple[set[int], dict[int, dict[str, Any]]]:
    """Validate document IDs, titles and abstract structures."""

    document_ids: set[int] = set()
    document_lookup: dict[int, dict[str, Any]] = {}

    for record in corpus:

        doc_id = record.get("doc_id")
        title = record.get("title")
        abstract = record.get("abstract")

        if (
            not isinstance(doc_id, int)
            or isinstance(doc_id, bool)
        ):
            raise SciFactValidationError(
                f"Invalid corpus document ID: {doc_id}"
            )

        if doc_id in document_ids:
            raise SciFactValidationError(
                f"Duplicate corpus document ID: {doc_id}"
            )

        if not isinstance(title, str):
            raise SciFactValidationError(
                f"Document {doc_id} has an invalid title."
            )

        if not isinstance(abstract, list):
            raise SciFactValidationError(
                f"Document {doc_id} has an invalid abstract."
            )

        if not all(
            isinstance(sentence, str)
            for sentence in abstract
        ):
            raise SciFactValidationError(
                f"Document {doc_id} contains a non-text abstract sentence."
            )

        document_ids.add(doc_id)
        document_lookup[doc_id] = record

    return document_ids, document_lookup


# ==================================================
# 4. VALIDATE CLAIMS AND EVIDENCE
# ==================================================

def validate_claims(
    claims: list[dict[str, Any]],
    document_ids: set[int],
    document_lookup: dict[int, dict[str, Any]],
    split_name: str,
    allow_unlabeled: bool = False,
) -> dict[str, Any]:
    """Validate claim IDs, evidence references and annotation labels."""

    seen_claim_ids: set[int] = set()

    evidence_count = 0
    empty_evidence_count = 0
    unlabeled_count = 0

    label_counts = {
        label: 0 for label in VALID_LABELS
    }

    for claim in claims:

        claim_id = claim.get("id")
        claim_text = claim.get("claim")

        # ------------------------------------------
        # Validate claim ID
        # ------------------------------------------

        if (
            not isinstance(claim_id, int)
            or isinstance(claim_id, bool)
        ):
            raise SciFactValidationError(
                f"Invalid claim ID in {split_name}: {claim_id}"
            )

        if claim_id in seen_claim_ids:
            raise SciFactValidationError(
                f"Duplicate claim ID {claim_id} in {split_name}."
            )

        seen_claim_ids.add(claim_id)

        # ------------------------------------------
        # Validate claim text
        # ------------------------------------------

        if (
            not isinstance(claim_text, str)
            or not claim_text.strip()
        ):
            raise SciFactValidationError(
                f"Claim {claim_id} has invalid claim text."
            )

        # ------------------------------------------
        # Handle unlabeled test records
        # ------------------------------------------

        has_evidence = "evidence" in claim
        has_cited_docs = "cited_doc_ids" in claim

        if not has_evidence and not has_cited_docs:

            if allow_unlabeled:
                unlabeled_count += 1
                continue

            raise SciFactValidationError(
                f"Claim {claim_id} in {split_name} "
                "is missing evidence annotations."
            )

        if has_evidence != has_cited_docs:
            raise SciFactValidationError(
                f"Claim {claim_id} has incomplete annotation fields."
            )

        # ------------------------------------------
        # Validate evidence structure
        # ------------------------------------------

        evidence = claim["evidence"]
        cited_doc_ids = claim["cited_doc_ids"]

        if not isinstance(evidence, dict):
            raise SciFactValidationError(
                f"Claim {claim_id} has invalid evidence structure."
            )

        if not isinstance(cited_doc_ids, list):
            raise SciFactValidationError(
                f"Claim {claim_id} has invalid cited_doc_ids."
            )

        for cited_id in cited_doc_ids:

            if (
                not isinstance(cited_id, int)
                or isinstance(cited_id, bool)
            ):
                raise SciFactValidationError(
                    f"Claim {claim_id} has an invalid cited document ID."
                )

            if cited_id not in document_ids:
                raise SciFactValidationError(
                    f"Claim {claim_id} cites missing document {cited_id}."
                )

        # Empty evidence is allowed in labeled records.
        if not evidence:
            empty_evidence_count += 1

        # ------------------------------------------
        # Validate evidence documents
        # ------------------------------------------

        for doc_id_text, annotations in evidence.items():

            if not isinstance(doc_id_text, str):
                raise SciFactValidationError(
                    f"Claim {claim_id} has a non-string evidence document ID."
                )

            try:
                doc_id = int(doc_id_text)

            except ValueError as exc:
                raise SciFactValidationError(
                    f"Claim {claim_id} has invalid evidence document ID."
                ) from exc

            if str(doc_id) != doc_id_text:
                raise SciFactValidationError(
                    f"Claim {claim_id} has a non-standard document ID: "
                    f"{doc_id_text}"
                )

            if doc_id not in document_ids:
                raise SciFactValidationError(
                    f"Claim {claim_id} references missing document {doc_id}."
                )

            if doc_id not in cited_doc_ids:
                raise SciFactValidationError(
                    f"Claim {claim_id} has evidence document {doc_id} "
                    "absent from cited_doc_ids."
                )

            if not isinstance(annotations, list) or not annotations:
                raise SciFactValidationError(
                    f"Claim {claim_id} has invalid annotations "
                    f"for document {doc_id}."
                )

            abstract_length = len(
                document_lookup[doc_id]["abstract"]
            )

            # --------------------------------------
            # Validate individual annotations
            # --------------------------------------

            for annotation in annotations:

                if not isinstance(annotation, dict):
                    raise SciFactValidationError(
                        f"Claim {claim_id} has an invalid annotation."
                    )

                label = annotation.get("label")

                if (
                    not isinstance(label, str)
                    or label not in VALID_LABELS
                ):
                    raise SciFactValidationError(
                        f"Claim {claim_id} has an unknown label: {label}"
                    )

                label_counts[label] += 1

                sentences = annotation.get("sentences")

                if not isinstance(sentences, list) or not sentences:
                    raise SciFactValidationError(
                        f"Claim {claim_id} has invalid sentence indices."
                    )

                for sentence_index in sentences:

                    if (
                        not isinstance(sentence_index, int)
                        or isinstance(sentence_index, bool)
                        or sentence_index < 0
                    ):
                        raise SciFactValidationError(
                            f"Claim {claim_id} has an invalid sentence index."
                        )

                    if sentence_index >= abstract_length:
                        raise SciFactValidationError(
                            f"Claim {claim_id} references sentence "
                            f"{sentence_index} outside document {doc_id}."
                        )

                evidence_count += 1

    return {
        "claims": len(claims),
        "evidence_annotations": evidence_count,
        "claims_without_evidence": empty_evidence_count,
        "unlabeled_claims": unlabeled_count,
        "support": label_counts["SUPPORT"],
        "contradict": label_counts["CONTRADICT"],
        "not_enough_info": label_counts["NOT_ENOUGH_INFO"],
    }


# ==================================================
# 5. MAIN VALIDATION PIPELINE
# ==================================================

def main() -> int:

    print("=" * 55)
    print("       SCIENTIFIC FACT CHECKING DATASET")
    print("              SCIFACT VALIDATION")
    print("=" * 55)

    print(f"\nDataset location: {DATA_DIR}\n")

    # Load and validate corpus
    corpus = load_jsonl(
        DATA_DIR / FILES["corpus"]
    )

    document_ids, document_lookup = validate_corpus(corpus)

    print(f"Corpus documents: {len(document_ids):,}")
    print("Corpus document IDs: unique")
    print("Corpus abstract structures: valid\n")

    total_claims = 0
    total_evidence = 0
    total_empty = 0
    total_unlabeled = 0

    total_labels = {
        label: 0 for label in VALID_LABELS
    }

    # Process all three splits
    for split in ("train", "dev", "test"):

        filename = FILES[split]

        print(f"\nProcessing {filename}...")

        claims = load_jsonl(
            DATA_DIR / filename
        )

        summary = validate_claims(
            claims=claims,
            document_ids=document_ids,
            document_lookup=document_lookup,
            split_name=split,
            allow_unlabeled=(split == "test"),
        )

        total_claims += summary["claims"]
        total_evidence += summary["evidence_annotations"]
        total_empty += summary["claims_without_evidence"]
        total_unlabeled += summary["unlabeled_claims"]

        total_labels["SUPPORT"] += summary["support"]
        total_labels["CONTRADICT"] += summary["contradict"]
        total_labels["NOT_ENOUGH_INFO"] += summary["not_enough_info"]

        print("-" * 55)
        print(f"FILE: {filename}")
        print(f"Claims: {summary['claims']:,}")
        print(
            f"Evidence annotations: "
            f"{summary['evidence_annotations']:,}"
        )
        print(
            f"Claims without evidence: "
            f"{summary['claims_without_evidence']:,}"
        )
        print(
            f"Unlabeled claims: "
            f"{summary['unlabeled_claims']:,}"
        )
        print("Claim IDs: unique within split")
        print("Document references: valid")
        print("Evidence sentence indices: valid")
        print("Evidence labels: valid for annotated records")

    # ----------------------------------------------
    # Final summary
    # ----------------------------------------------

    print("\n" + "=" * 55)
    print("                 FINAL SUMMARY")
    print("=" * 55)

    print(f"Total corpus documents: {len(document_ids):,}")
    print(f"Total claims: {total_claims:,}")
    print(f"Total evidence annotations: {total_evidence:,}")
    print(f"Claims without evidence: {total_empty:,}")
    print(f"Unlabeled claims: {total_unlabeled:,}")

    print("\nEvidence label distribution:")

    for label, count in total_labels.items():
        print(f"  {label}: {count:,}")

    print("\nAll SciFact validation checks passed.")

    return 0


# ==================================================
# 6. SCRIPT ENTRY POINT
# ==================================================

if __name__ == "__main__":

    try:
        raise SystemExit(main())

    except SciFactValidationError as exc:

        print(f"\nSCIFACT VALIDATION FAILED: {exc}")

        raise SystemExit(1)

