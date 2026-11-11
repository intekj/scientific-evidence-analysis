
import json
from pathlib import Path

PREPARED_DIR = (
    Path(__file__).resolve().parent
    / "datasets"
    / "benchmarks"
    / "scifact"
    / "prepared"
)

FILES = [
    "claims_train_prepared.jsonl",
    "claims_dev_prepared.jsonl",
    "claims_test_prepared.jsonl",
]


def main():
    for filename in FILES:
        file_path = PREPARED_DIR / filename

        print("\n" + "=" * 60)
        print(filename)
        print("=" * 60)

        records = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))

        print("Records:", len(records))

        statuses = {}

        for record in records:
            status = record["evidence_status"]
            statuses[status] = statuses.get(status, 0) + 1

        print("Evidence statuses:", statuses)

        if records:
            print("\nFirst prepared record:")
            print(json.dumps(
                records[0],
                indent=4,
                ensure_ascii=False
            )[:1800])

    print("\nPrepared dataset inspection completed.")


if __name__ == "__main__":
    main()
