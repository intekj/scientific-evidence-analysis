
import json
from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parent
    / "datasets"
    / "benchmarks"
    / "scifact"
    / "data"
    / "claims_train.jsonl"
)


def load_claims():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    claims = load_claims()

    examples = {
        "SUPPORT": None,
        "CONTRADICT": None,
        "NO_EVIDENCE": None,
    }

    for claim in claims:
        evidence = claim.get("evidence", {})

        if not evidence:
            if examples["NO_EVIDENCE"] is None:
                examples["NO_EVIDENCE"] = claim

        for annotations in evidence.values():
            for annotation in annotations:
                label = annotation.get("label")

                if label in examples and examples[label] is None:
                    examples[label] = claim

    for category, example in examples.items():
        print("\n" + "=" * 65)
        print(category)
        print("=" * 65)

        if example is None:
            print("No example found.")
        else:
            print(json.dumps(example, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
