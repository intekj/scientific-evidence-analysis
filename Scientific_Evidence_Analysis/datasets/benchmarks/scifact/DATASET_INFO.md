
# SciFact Dataset Documentation

## 1. Dataset Overview

**Dataset:** SciFact  
**Purpose:** Scientific claim verification and evidence identification.

SciFact is used in this project as a benchmark for evaluating scientific claim verification and claim-evidence relationships.

It provides a collection of scientific documents and claims with evidence annotations in the training and development splits.

## 2. Dataset Location

```text
datasets/benchmarks/scifact/
├── data/
│   ├── corpus.jsonl
│   ├── claims_train.jsonl
│   ├── claims_dev.jsonl
│   └── claims_test.jsonl
├── manifest.json
└── DATASET_INFO.md
```

## 3. Dataset Statistics

| Component | Records |
|---|---:|
| Corpus documents | 5,183 |
| Training claims | 809 |
| Development claims | 300 |
| Test claims | 300 |
| Total claims | 1,409 |
| Training evidence annotations | 957 |
| Development evidence annotations | 338 |
| Total evidence annotations | 1,295 |

## 4. Evidence Label Distribution

The following labels were found in the annotated records:

| Label | Count |
|---|---:|
| SUPPORT | 832 |
| CONTRADICT | 463 |
| NOT_ENOUGH_INFO | 0 |
| **Total** | **1,295** |

The original label terminology is preserved.

## 5. Split Characteristics

### Training Split

- 809 claims.
- 957 evidence annotations.
- 304 claims without evidence annotations.
- No unlabeled claims detected.

### Development Split

- 300 claims.
- 338 evidence annotations.
- 112 claims without evidence annotations.
- No unlabeled claims detected.

### Test Split

- 300 claims.
- No evidence annotations available in the downloaded records.
- All 300 claims are unlabeled.

**Important:** The test split must not be treated as a labeled dataset for calculating supervised claim-verification accuracy.

## 6. Data Structure

### Corpus Records

The corpus contains scientific document records with fields such as:

- `doc_id`: Document identifier.
- `title`: Scientific paper title.
- `abstract`: Abstract represented as a list of text segments.
- `structured`: Indicates whether the abstract is structured.

### Claim Records

Claim records contain:

- `id`: Claim identifier.
- `claim`: Scientific statement to verify.
- `evidence`: Evidence annotations grouped by document identifier, when available.
- `cited_doc_ids`: Associated document identifiers, when available.

Evidence annotations include:

- Evidence sentence indices.
- Evidence labels.
- Document references.

## 7. Usage in This Research Project

SciFact will support the following experimental components:

1. Scientific claim verification.
2. Supporting and contradicting evidence identification.
3. Claim-evidence alignment.
4. Evidence retrieval evaluation.
5. Comparison of the proposed framework with benchmark-based approaches.

SciFact will be used alongside the project's domain-specific scientific paper collection.

## 8. Dataset Limitations

1. The downloaded test split contains 300 claims without gold evidence annotations.
2. The annotated records in the training and development splits contain SUPPORT and CONTRADICT labels only.
3. The validation results establish structural consistency, not scientific correctness of the annotations.
4. SciFact does not, by itself, provide the full context fields required for all methodology-aware disagreement experiments.
5. Additional domain-specific annotation will be required to evaluate whether differences between scientific findings are genuine contradictions or context-dependent differences.

## 9. Validation

The dataset was checked using `prepare_scifact.py`.

Validation included:

- JSONL parsing.
- Unique corpus document identifiers.
- Unique claim identifiers within each split.
- Valid abstract structures.
- Valid document references.
- Valid evidence sentence indices.
- Valid evidence labels for annotated records.
- Detection of unlabeled claims.

All implemented SciFact validation checks passed.

## 10. Reproducibility

The file `manifest.json` records:

- Dataset file sizes.
- SHA-256 hashes.
- Record counts.
- Evidence annotation counts.
- Evidence label distribution.
- Split-level statistics.

The manifest should be regenerated whenever the underlying dataset files change.
