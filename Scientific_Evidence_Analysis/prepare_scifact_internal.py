
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "benchmarks"
    / "scifact"
    / "data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "benchmarks"
    / "scifact"
    / "prepared"
)


def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {file_path}, line {line_number}"
                    ) from error

    return records


def build_document_lookup():
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")

    return {
        str(document["doc_id"]): document
        for document in corpus
    }


def normalize_claim(claim, split, document_lookup):
    has_evidence_field = "evidence" in claim
    has_cited_field = "cited_doc_ids" in claim

    if not has_evidence_field and not has_cited_field:
        annotation_available = False
        evidence_status = "GOLD_ANNOTATION_UNAVAILABLE"
        evidence = []

    elif has_evidence_field and has_cited_field:
        annotation_available = True
        evidence = []

        for doc_id, annotations in claim["evidence"].items():
            doc_id = str(doc_id)

            if doc_id not in document_lookup:
                raise ValueError(
                    f"Claim {claim['id']} references missing document {doc_id}"
                )

            document = document_lookup[doc_id]
            abstract = document.get("abstract", [])

            for annotation in annotations:
                sentence_indices = annotation["sentences"]

                evidence_sentences = []

                for sentence_index in sentence_indices:
                    if not isinstance(sentence_index, int):
                        raise ValueError(
                            f"Invalid sentence index in claim {claim['id']}"
                        )

                    if sentence_index < 0 or sentence_index >= len(abstract):
                        raise ValueError(
                            f"Sentence index {sentence_index} is out of range "
                            f"for document {doc_id}"
                        )

                    evidence_sentences.append(abstract[sentence_index])

                evidence.append({
                    "document_id": doc_id,
                    "document_title": document.get("title"),
                    "sentence_indices": sentence_indices,
                    "evidence_sentences": evidence_sentences,
                    "label": annotation["label"],
                })

        evidence_status = (
            "ANNOTATED_WITH_EVIDENCE"
            if evidence
            else "ANNOTATED_NO_EVIDENCE"
        )

    else:
        raise ValueError(
            f"Claim {claim['id']} has incomplete annotation fields"
        )

    return {
        "dataset": "SciFact",
        "split": split,
        "claim_id": claim["id"],
        "claim_text": claim["claim"],
        "cited_doc_ids": claim.get("cited_doc_ids", []),
        "annotation_available": annotation_available,
        "evidence_status": evidence_status,
        "evidence": evidence,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document_lookup = build_document_lookup()

    print(f"Loaded {len(document_lookup)} corpus documents.")

    total_claims = 0
    total_evidence = 0

    for split in ["train", "dev", "test"]:
        input_file = DATA_DIR / f"claims_{split}.jsonl"
        output_file = OUTPUT_DIR / f"claims_{split}_prepared.jsonl"

        claims = load_jsonl(input_file)

        prepared_records = [
            normalize_claim(claim, split, document_lookup)
            for claim in claims
        ]

        with open(output_file, "w", encoding="utf-8") as file:
            for record in prepared_records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        evidence_count = sum(
            len(record["evidence"])
            for record in prepared_records
        )

        total_claims += len(prepared_records)
        total_evidence += evidence_count

        print(f"\n{split.upper()} split")
        print(f"Claims prepared: {len(prepared_records)}")
        print(f"Evidence records: {evidence_count}")
        print(f"Output: {output_file}")

    print("\n" + "=" * 50)
    print("SCIFACT INTERNAL DATASET PREPARATION COMPLETE")
    print("=" * 50)
    print(f"Total claims: {total_claims}")
    print(f"Total evidence records: {total_evidence}")


if __name__ == "__main__":
    main()
