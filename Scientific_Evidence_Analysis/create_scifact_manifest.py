
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "datasets" / "benchmarks" / "scifact" / "data"
OUTPUT_FILE = PROJECT_ROOT / "datasets" / "benchmarks" / "scifact" / "manifest.json"

FILES = [
    "corpus.jsonl",
    "claims_train.jsonl",
    "claims_dev.jsonl",
    "claims_test.jsonl",
]


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def read_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    manifest = {
        "dataset": "SciFact",
        "description": "Scientific claim verification dataset",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "statistics": {},
    }

    # Check and record the four dataset files
    for filename in FILES:
        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {file_path}")

        manifest["files"][filename] = {
            "size_bytes": file_path.stat().st_size,
            "sha256": calculate_sha256(file_path),
        }

    # Corpus statistics
    corpus = read_jsonl(DATA_DIR / "corpus.jsonl")

    manifest["statistics"]["corpus_documents"] = len(corpus)

    # Claim split statistics
    total_claims = 0
    total_annotations = 0
    total_without_evidence = 0
    total_unlabeled = 0
    label_counts = Counter()
    split_statistics = {}

    for split in ["train", "dev", "test"]:
        claims = read_jsonl(DATA_DIR / f"claims_{split}.jsonl")

        annotation_count = 0
        without_evidence = 0
        unlabeled = 0

        for claim in claims:
            if "evidence" not in claim and "cited_doc_ids" not in claim:
                unlabeled += 1
                continue

            evidence = claim.get("evidence", {})

            if not evidence:
                without_evidence += 1

            for document_annotations in evidence.values():
                for annotation in document_annotations:
                    annotation_count += 1
                    label_counts[annotation["label"]] += 1

        split_statistics[split] = {
            "claims": len(claims),
            "evidence_annotations": annotation_count,
            "claims_without_evidence": without_evidence,
            "unlabeled_claims": unlabeled,
        }

        total_claims += len(claims)
        total_annotations += annotation_count
        total_without_evidence += without_evidence
        total_unlabeled += unlabeled

    manifest["statistics"]["claim_splits"] = split_statistics
    manifest["statistics"]["total_claims"] = total_claims
    manifest["statistics"]["total_evidence_annotations"] = total_annotations
    manifest["statistics"]["claims_without_evidence"] = total_without_evidence
    manifest["statistics"]["unlabeled_claims"] = total_unlabeled
    manifest["statistics"]["evidence_label_distribution"] = dict(label_counts)

    # Save manifest
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4, ensure_ascii=False)

    print("SciFact manifest created successfully.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Corpus documents: {len(corpus)}")
    print(f"Total claims: {total_claims}")
    print(f"Evidence annotations: {total_annotations}")
    print(f"Label distribution: {dict(label_counts)}")


if __name__ == "__main__":
    main()
