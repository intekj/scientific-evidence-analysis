
from datasets import load_from_disk
from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parent
    / "datasets"
    / "benchmarks"
    / "qasper"
    / "data"
    / "prepared"
)


def main():
    dataset = load_from_disk(str(DATA_DIR))

    total_questions = 0

    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]

        question_count = 0

        for paper in split:
            qas = paper.get("qas")

            if not qas:
                continue

            if isinstance(qas, dict):
                questions = qas.get("question", [])

            elif isinstance(qas, list):
                questions = [
                    item.get("question")
                    for item in qas
                    if isinstance(item, dict)
                ]

            else:
                raise ValueError(
                    f"Unexpected QAS structure in {split_name}"
                )

            if isinstance(questions, list):
                question_count += len(questions)
            elif questions:
                question_count += 1

        total_questions += question_count

        print(f"\n{split_name.upper()}")
        print("Papers:", len(split))
        print("Questions:", question_count)

    print("\n" + "=" * 45)
    print("TOTAL QUESTIONS:", total_questions)
    print("=" * 45)


if __name__ == "__main__":
    main()
