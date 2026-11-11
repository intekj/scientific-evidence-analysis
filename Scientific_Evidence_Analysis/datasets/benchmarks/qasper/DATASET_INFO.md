# QASPER Dataset

## Name and purpose

QASPER (Question Answering on Scientific Papers) is an English-language NLP
benchmark for information-seeking question answering over full scientific
research papers. Questions are written from a paper's title and abstract and
answered using evidence from the full paper.

The official dataset card describes 5,049 questions over 1,585 Natural Language
Processing papers. The downloaded data was inspected locally; its observed split
counts are listed below.

## Official source, licence, and use

- Hugging Face dataset: [allenai/qasper](https://huggingface.co/datasets/allenai/qasper)
- Official dataset card and schema:
  [Hugging Face README](https://huggingface.co/datasets/allenai/qasper#dataset-card-for-qasper)
- Project page: [Allen Institute QASPER](https://allenai.org/data/qasper)
- Dataset paper: [Dasigi et al. (2021), arXiv:2105.03011](https://arxiv.org/abs/2105.03011)
- Licence declared in the official Hugging Face dataset card: **CC BY 4.0**
  ([licence text](https://creativecommons.org/licenses/by/4.0/)).

Redistribution and adaptations must follow CC BY 4.0, including appropriate
attribution and indication of changes. The card does not provide a separate
per-paper rights analysis for the source full texts; review those rights before
redistributing derived or extracted paper content.

## Download method and local layout

The repository's current `main` tree contains a legacy `qasper.py` loader and a
small dummy archive, not the full corpus as ordinary repository files. The
official Hugging Face datasets viewer lists the current full train, validation,
and test data as converted Parquet files. The project downloader retrieves those
official Parquet URLs from the viewer API and loads them with the supported
Hugging Face `datasets` Parquet builder (`load_dataset("parquet", data_files=...)`).
This avoids relying on the legacy dataset loading script.

Run from the project root using the project Python environment:

```powershell
.\venv\Scripts\python.exe download_qasper.py
.\venv\Scripts\python.exe inspect_qasper.py
```

Files are stored under `datasets/benchmarks/qasper/data/`:

- `raw/` — the original downloaded official split Parquet files.
- `prepared/` — a Hugging Face `DatasetDict` saved with `save_to_disk`, for
  offline reload via `datasets.load_from_disk`.
- `download_manifest.json` — source URLs, expected file sizes, and observed
  record counts.

The downloader reuses size- and Parquet-validated raw files and a complete
prepared dataset on reruns. Downloads are written to temporary files and promoted
only after validation. It does not fetch or mix the SciFact dataset.

## Observed splits and counts

Counts below were measured from the downloaded local dataset by
`inspect_qasper.py`; answerability counts are per question.

| Split | Paper records | Unique papers | Questions | Answerable annotations | Unanswerable annotations | Mixed annotations |
|---|---:|---:|---:|---:|---:|---:|
| Train | 888 | 888 | 2,593 | 2,315 | 271 | 7 |
| Validation | 281 | 281 | 1,005 | 870 | 60 | 75 |
| Test | 416 | 416 | 1,451 | 1,208 | 79 | 164 |
| **Total** | **1,585** | **1,585** | **5,049** | **4,393** | **410** | **246** |

No questions in the inspected data had unspecified answerability. A question is
counted as mixed when its annotations include both `unanswerable=true` and
`unanswerable=false`; mixed questions are shown separately rather than forced
into either answerable count.

## Available fields

The observed Hugging Face feature schema has six top-level fields:

- `id` — paper identifier (the Paper ID).
- `title` — paper title.
- `abstract` — abstract text; it may be empty or absent in individual records.
- `full_text` — an ordered sequence of sections. Each section has
  `section_name` and `paragraphs`; each paragraph is text.
- `qas` — questions aligned with question IDs and question-writer metadata.
  Available members include `question`, `question_id`, `nlp_background`,
  `topic_background`, `paper_read`, `search_query`, `question_writer`, and
  `answers`.
- `figures_and_tables` — sequence of figure/table captions and associated file
  references, where present.

Each question's answer annotations can include:

- `unanswerable` — answerability annotation.
- `extractive_spans`, `yes_no`, and `free_form_answer` — answer forms.
- `evidence` — supporting paragraph, figure, or table passages. Figure/table
  selections may use the `FLOAT SELECTED` marker.
- `highlighted_evidence` — highlighted supporting sentences where supplied.
- `annotation_id` and `worker_id` — annotation provenance identifiers.

Nested values and optional fields can be empty or missing. Consumers should check
the field and annotation content rather than assume every question has an answer
or evidence passage.

## Five actual sample observations

The following are five of the first records inspected from the downloaded test
split. Question wording and answer/evidence details are from the local dataset;
passage text is abbreviated here for readability.

1. **1911.10742 — “End-to-End Trainable Non-Collaborative Dialog System”**
   Question: “How big is the ANTISCAM dataset?” Answer marked answerable; an
   extractive span reports “3,044 sentences in 100 dialogs.” Evidence points to
   the paragraph describing collection and annotation of the AntiScam corpus.
2. **1904.09131 — “OpenTapioca: Lightweight Entity Linking for Wikidata”**
   Question: “What is the accuracy of this model compared to sota?” The answer
   annotations are mixed; the evidence includes a `FLOAT SELECTED` reference to
   Figure 2 (F1 scores on test datasets).
3. **1611.06322 — “Spotting Rumors via Novelty Detection”**
   Question: “What previous methods do they compare against?” Answer marked
   answerable; extractive spans name the Liu et al. (2015) and Yang et al. (2012)
   early-rumour-detection baselines.
4. **1604.02038 — “Sentence Level Recurrent Topic Model: Letting Topics Speak
   for Themselves”**
   Question: “What baselines did they compare with?” Answer marked answerable;
   extractive spans include LDA, Doc-NADE, and HTMM.
5. **1911.04474 — “TENER: Adapting Transformer Encoder for Named Entity
   Recognition”**
   Question: “Which NER dataset do they use?” Answer marked answerable; extractive
   spans include CoNLL2003 and OntoNotes versions 5.0 and 4.0.

## Limitations

- QASPER is centered on NLP research papers; its subject matter does not represent
  flood susceptibility research specifically.
- Questions and answers are annotations by NLP practitioners, not necessarily
  expert domain reviewers; annotations can differ, as mixed answerability
  demonstrates.
- The full text is represented as extracted sections and paragraph strings;
  layout, figures, tables, and other visual structure may not be fully preserved.
- An evidence value may refer to a paragraph, figure, or table; `FLOAT SELECTED`
  entries are not ordinary paragraph text.
- Full-text availability and metadata completeness vary by record.
- A benchmark score measures a defined benchmark task and does not by itself
  establish scientific correctness or retrieval quality for another domain.

## Relevance to Scientific Evidence Analysis

QASPER can support controlled evaluation of:

- Full-paper scientific question answering.
- Retrieval of evidence passages from research papers.
- Aggregation of information across multiple paper sections.
- Answer generation that is grounded in supporting evidence.
- Citation/provenance-aware evaluation of evidence-grounded research responses.

It is a general **NLP scientific-paper benchmark**, not a flood-susceptibility
dataset. It can evaluate the project's scientific question answering and evidence
retrieval workflow, but it cannot validate flood-specific findings or replace
evaluation on domain-relevant flood literature.

## Citation

```text
Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., & Gardner, M. (2021).
A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers.
Proceedings of NAACL-HLT 2021. https://arxiv.org/abs/2105.03011
```
