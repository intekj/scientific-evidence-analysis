# Scientific Evidence Analysis

**A Multimodal Context-Aware Framework for Scientific Evidence Alignment, Claim
Verification, and Disagreement-Aware Synthesis**

A local Streamlit prototype for uploading scientific PDFs, retrieving relevant
passages across papers, reviewing preliminary claim/evidence signals, comparing
study context, and exporting a traceable evidence summary.

## Objectives

- Extract selectable PDF text and basic metadata with page references.
- Retrieve relevant passages across all uploaded papers.
- Group passages into supporting, contradicting, and neutral/insufficient
  categories using visible, rule-based directional cues.
- Compare passage context to flag potential differences in dataset, population,
  location, conditions, methods, models, metrics, and study period.
- Produce a balanced report whose extracted evidence findings cite a paper and page.

This is a research-support prototype, not an automated judge of scientific truth.
Similarity and keyword classifications must be checked by a human against each source.

## Technologies

- Python 3.10+
- Streamlit
- PyMuPDF (`fitz`)
- Pandas
- scikit-learn TF-IDF and cosine similarity
- Optional Sentence Transformers `all-MiniLM-L6-v2` on CPU when the model is
  already cached locally

The standard installation does not install PyTorch or download a model. TF-IDF
works locally without a GPU, network access, API keys, or model files.

## Project structure

```text
Scientific_Evidence_Analysis/
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   ├── embedding_engine.py
│   ├── evidence_retrieval.py
│   ├── claim_verification.py
│   ├── context_extraction.py
│   ├── disagreement_analysis.py
│   ├── evidence_synthesis.py
│   ├── curated_dataset.py
│   ├── report_export.py
│   └── paper_storage.py
├── data/
│   ├── papers.json
│   ├── uploads/
│   └── processed/
├── models/
├── assets/
└── tests/
```

Original PDFs are stored locally in `data/uploads/`; their SHA-256 keyed metadata
is stored in `data/papers.json`. At application startup, available PDFs are
re-extracted to restore page text and chunks for retrieval, verification,
disagreement analysis, and synthesis. Extracted text and chunks are regenerated
in memory rather than persisted separately. Files and metadata stay in the project
directory and are not sent to an external service.

The app also discovers the 21 PDFs named by the canonical P001–P021 mapping in
`modules/curated_dataset.py` under `../AI Papers Dataset/papers/` and copies them
through the same local PDF store and extraction pipeline. Only those exact mapped
filenames are imported; other PDFs in that directory are ignored. Canonical IDs
are stored as additive metadata while the store retains its SHA-256 identity,
and full-text chunks expose the canonical ID, mapped PDF filename, and original
1-based PDF page number. Existing user uploads continue to use their SHA-256
paper IDs. Canonical source PDFs are never modified. Already registered canonical
papers are restored from local storage rather than copied again; missing,
unreadable, or textless mapped files are reported in the app.

## Curated master-claims pipeline

When present, the app loads the master claim CSV from the sibling dataset
directory, resolved relative to this project (not the process working directory):

```text
../AI Papers Dataset/Master Dataset/claims.csv
```

The loader validates required columns, nonempty claim identity/text fields,
duplicate claim IDs, the P001–P021 canonical paper IDs, and the one-to-one
`paper_id` / `source_filename` mapping. It reads the CSV as text, retains every
source column unchanged in memory, and adds normalized review-status fields; it
never edits the source file. Missing evidence quotations or page fields are
allowed and displayed as unavailable rather than inferred.

Validation decisions are normalized conservatively to `RETAINED`, `CORRECTED`,
`MERGED`, `REJECTED`, `CANDIDATE / NEEDS REVIEW`, `CONTEXT ONLY`, or
`UNKNOWN / NEEDS REVIEW`. Rejections and merges take precedence over other
decision wording; corrections remain distinct from retained claims. A record is
not marked retained based on a quotation alone: an affirmative decision and a
reviewed annotation status are both required. Original decision, annotation,
evidence-status, and rationale columns remain visible alongside the normalized
status.

On the **Evidence Retrieval** page, curated claim retrieval searches claim text
and verified evidence quotations (using the evidence text field when no verified
quotation is present). It uses its own `EmbeddingEngine` index and displays
curated claims and source quotations separately from uploaded-PDF passage
retrieval. Scores from these two indexes are not combined and curated claim
matches are not treated as full-text PDF passage evidence. Curated citations use
the dataset's canonical paper ID, source filename, evidence page, and section.

Optional cross-paper data is loaded independently from:

```text
../AI Papers Dataset/Cross Paper Analysis/Batch12_Cross_Paper_Comparison.csv
../AI Papers Dataset/Cross Paper Analysis/Batch12B_Disagreement_Analysis.csv
../AI Papers Dataset/Cross Paper Analysis/Batch12C_Uncertainty_Register.csv
```

Each optional CSV is schema-checked independently. The app preserves its
relationship/classification and uncertainty text, then shows records only when
their explicit claim IDs or canonical paper IDs link to the current curated
search results. Missing or invalid optional files generate a warning without
preventing the app from starting. A missing or invalid master claims file is
reported; uploaded-PDF search remains available.

These are source-curated review decisions and evidence locators, not an
automated scientific verdict. PDF-based labels remain preliminary heuristic
signals; similarity is not entailment, and cross-paper relation labels are
displayed as recorded rather than newly inferred. The analysis is limited to
the dataset's current paper coverage and its manually curated quotations,
page references, and review states.

## Installation (Windows / PowerShell)

Open PowerShell in the project folder. The project includes a `venv`; activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the interpreter directly for the commands
below, or enable script activation for the current user according to your local
IT policy. Install core dependencies:

```powershell
python -m pip install -r requirements.txt
```

The `venv` must use Python 3.10 or newer. No GPU toolkit is required.

### Optional cached semantic model

If you already have `sentence-transformers` and the model
`sentence-transformers/all-MiniLM-L6-v2` in the local Hugging Face cache, the app
uses it on CPU. It does not attempt to download a model automatically. To prepare
the optional model yourself, install the optional package and download the model
using an explicitly chosen workflow; the core TF-IDF fallback remains available.

## Run

```powershell
streamlit run app.py
```

Open the local address printed by Streamlit, usually
`http://localhost:8501`.

## How to use

1. **Upload Research Papers:** select one or more PDFs and click **Process selected
   PDFs**. The original PDF is saved locally, and the collection is restored
   automatically on later visits/restarts. Review each paper's filename, page
   count, processing status, upload date, and extracted text preview. Remove
   selected papers or confirm **Clear complete paper collection** to remove all
   locally stored papers. The 21 mapped canonical PDFs are also added
   automatically when available; the page reports any missing or unreadable files.
2. **Evidence Retrieval:** enter a scientific claim or question and search. Each
   passage shows source filename, page, and similarity score.
3. **Claim Verification:** retrieve passages and inspect their preliminary
   supporting, contradicting, or neutral grouping and the reason for each.
4. **Disagreement Analysis:** select one passage from each of two papers. The
   system shows context clues, fields that differ, and a cautious category.
5. **Evidence Synthesis:** after claim verification, review the structured report
   and download it as Markdown.
6. **About the Project:** review the architecture, methodology, and limitations.

## Method and interpretation

Retrieval uses cosine similarity on a local cached MiniLM model if available;
otherwise it uses TF-IDF word n-grams. Claim grouping requires topical term overlap
plus simple directional language cues. If those cues are ambiguous, the passage is
neutral/insufficient rather than being forced into a support/contradiction label.

Retrieval boosts passages with results/findings/conclusion cues, discounts
literature-review wording, filters citation-dense reference entries, suppresses
exact and near-duplicate excerpts, and ensures leading results from different
papers are represented when available. Similarity scores remain the raw retrieval
scores; section cues only affect ordering.

Claim verification uses topical overlap and reported result cues rather than
polarity words anywhere in a passage. Questions are displayed as relevant retrieved
evidence without support/contradiction labels. Titles, author descriptions, and
methodology-only passages cannot establish a conflict. A different reported metric
value remains neutral unless comparable experimental settings are established.
An asserted model-ranking claim is compared against named-model metric values in
the passage; the highest reported value is not attributed to an individual model
when a named ensemble has a higher value.
Disagreement comparison extracts source-text snippets for dataset, geographic
location, methodology, models, training/validation, input variables, metric names
and values, study period, and findings. The UI shows the source paper and page for
each field. Context differences plus distinct reported findings are labeled
**Context-Dependent Difference**; a different model ranking by itself is never
treated as a contradiction. A **Potential Genuine Contradiction** requires opposing
result cues and multiple matching extracted context fields. These categories are
cautious heuristics and cannot confirm that studies are genuinely comparable.
Geographic context ignores generic dataset names and filters explicit related-work
or citation wording; if no study location is stated in the passage, it remains
unidentified. Reported findings require an observed result or model/metric
comparison, rather than a data-collection period or method intention.
Question-oriented synthesis summarizes retrieved findings independently of their
claim-polarity label, separates individual-model and ensemble passages, and
aggregates study context across all uploaded pages for each source paper. Context
and finding statements retain their source filename and page citations.
Synthesis is template-based and uses only retrieved passage text and source/page
references; it does not generate uncited paper findings or authors.

## Limitations

- Scanned/image-only PDFs are rejected with an explanation; OCR is not included.
- Text-based table cells and figure captions are retained when PyMuPDF can extract
  them as page text. Figures and chart images are not visually interpreted.
- Metadata depends on PDF metadata fields and may be absent or inaccurate.
- Context extraction and result-direction labels are heuristics, not validated
  scientific NLI or causal inference.
- Similarity scores are not probabilities, evidence strength, or proof.
- PDFs and metadata persist under `data/`; extracted text and chunks are rebuilt
  from those PDFs when the application starts. Back up `data/` to preserve the
  local collection.
- PDF password recovery, DOI lookup, external APIs, and cloud services are not used.

## Future improvements

Optional OCR with explicit user consent, improved table parsing, human-reviewed
context forms, evaluation against annotated examples, and validated
domain-specific inference may be considered later.

## Tests

Install the optional test runner, then execute the suite:

```powershell
python -m pip install "pytest>=7,<9"
python -m pytest
```

Tests create temporary in-memory PDF content and synthetic passages to check
software behavior; they are not scientific evidence or benchmark findings.
