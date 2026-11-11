
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT / "datasets" / "benchmarks" / "scifact" / "data"
)

PREPARED_DIR = (
    PROJECT_ROOT / "datasets" / "benchmarks" / "scifact" / "prepared"
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    # Load original corpus
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")

    document_lookup = {
        str(document["doc_id"]): document
        for document in corpus
    }

    total_evidence = 0
    verified_evidence = 0
    errors = []

    # Check all labeled and prepared splits
    for split in ["train", "dev", "test"]:

        prepared_file = (
            PREPARED_DIR / f"claims_{split}_prepared.jsonl"
        )

        records = load_jsonl(prepared_file)

        print(f"\nChecking {split.upper()} split...")

        split_verified = 0
        split_evidence = 0

        for record in records:
            claim_id = record["claim_id"]

            for evidence in record["evidence"]:
                total_evidence += 1
                split_evidence += 1

                doc_id = evidence["document_id"]

                # Check document reference
                document = document_lookup.get(doc_id)

                if document is None:
                    errors.append(
                        f"Claim {claim_id}: missing document {doc_id}"
                    )
                    continue

                # Check document title
                if evidence["document_title"] != document.get("title"):
                    errors.append(
                        f"Claim {claim_id}: document title mismatch"
                    )
                    continue

                abstract = document.get("abstract", [])

                indices = evidence["sentence_indices"]
                sentences = evidence["evidence_sentences"]

                # Check sentence count
                if len(indices) != len(sentences):
                    errors.append(
                        f"Claim {claim_id}: sentence count mismatch"
                    )
                    continue

                # Check exact sentence text
                valid = True

                for index, sentence in zip(indices, sentences):

                    if not isinstance(index, int):
                        errors.append(
                            f"Claim {claim_id}: invalid sentence index"
                        )
                        valid = False
                        break

                    if index < 0 or index >= len(abstract):
                        errors.append(
                            f"Claim {claim_id}: sentence index out of range"
                        )
                        valid = False
                        break

                    if abstract[index] != sentence:
                        errors.append(
                            f"Claim {claim_id}: evidence text mismatch "
                            f"in document {doc_id}"
                        )
                        valid = False
                        break

                if not valid:
                    continue

                # Check original evidence label
                if evidence["label"] not in {
                    "SUPPORT",
                    "CONTRADICT",
                    "NOT_ENOUGH_INFO",
                }:
                    errors.append(
                        f"Claim {claim_id}: invalid evidence label"
                    )
                    continue

                split_verified += 1
                verified_evidence += 1

        print(f"Evidence records: {split_evidence}")
        print(f"Verified records: {split_verified}")

    print("\n" + "=" * 55)
    print("SCIFACT EVIDENCE PROVENANCE REPORT")
    print("=" * 55)

    print(f"Total evidence records: {total_evidence}")
    print(f"Verified evidence records: {verified_evidence}")
    print(f"Errors found: {len(errors)}")

    if errors:
        print("\nErrors:")
        for error in errors[:20]:
            print("-", error)

        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")

        raise SystemExit("Evidence provenance verification FAILED.")

    print("\nAll evidence provenance checks passed.")


if __name__ == "__main__":
    main()
