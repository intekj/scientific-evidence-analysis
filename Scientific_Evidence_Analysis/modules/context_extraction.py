"""Explainable regex-based extraction of study-context clues from passage text."""

from __future__ import annotations

import re

FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "Study population": (
        r"\bparticipants?\b", r"\bpatients?\b", r"\bsubjects?\b",
        r"\bchildren\b", r"\badults\b", r"\bcohort\b", r"\bpopulation\b",
    ),
    "Experimental conditions": (
        r"\b(?:condition|conditions|setting|settings|temperature|dose|duration|"
        r"environment|intervention|control group)\b",
    ),
    "Training and validation": (
        r"\b(?:train(?:ing)? set|test set|validation set|cross-validation|"
        r"hold[- ]out|split|folds?|k-fold|stratified|calibration|"
        r"training and validation|validation procedure)\b",
    ),
    "Input variables": (
        r"\b(?:input variables?|features?|predictors?|covariates?|"
        r"conditioning factors?|explanatory variables?|independent variables?)\b",
    ),
    "Methodology": (
        r"\b(?:methodology|method(?:s)?|approach|framework|model|algorithm|"
        r"machine learning|deep learning|supervised|unsupervised|ensemble|"
        r"voting|randomized|randomised|trial|cross-sectional|longitudinal|"
        r"observational|qualitative|quantitative|meta-analysis|systematic review|"
        r"experiment|ablation|compared with|comparison|validation|case study)\b",
    ),
    "Evaluation metrics": (
        r"\b(?:overall accuracy|accuracy|kappa(?: coefficient)?|precision|recall|"
        r"F1(?:[- ]score)?|AUC|area under (?:the )?curve|RMSE|MAE|sensitivity|"
        r"specificity|balanced accuracy|IoU|MCC|R(?:2|²))\b",
    ),
    "Study period": (
        r"\b(?:from\s+)?(?:19|20)\d{2}\s*(?:to|[-–])\s*(?:19|20)\d{2}\b",
        r"\b(?:between\s+)?(?:19|20)\d{2}\s*(?:and|to)\s*(?:19|20)\d{2}\b",
        r"\b(?:19|20)\d{2}\b",
    ),
    "Individual versus ensemble performance": (
        r"\b(?:ensemble|voting|stacking|bagging|boosting|individual model|"
        r"single model|base learner|best individual|best overall)\b",
    ),
}

MODEL_NAMES = (
    "Cascade Forest Model", "LightGBM", "XGBoost", "CatBoost", "Random Forest",
    "Decision Tree", "Gradient Boosting", "Support Vector Machine",
    "Gaussian Linear Model", "Deep Neural Network", "Neural Network",
    "Logistic Regression", "Linear Regression", "Ensemble Voting",
    "Voting Classifier", "Voting Ensemble", "Ensemble Model", "Ensemble", "GLM",
    "DNN", "SVM", "CNN", "LSTM", "RNN", "RF", "DT", "CFM",
)
MODEL_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(name) for name in sorted(MODEL_NAMES, key=len, reverse=True))
    + r")(?!\w)",
    re.IGNORECASE,
)

DATASET_TERMS = (
    "dataset", "data set", "database", "corpus", "benchmark", "inventory",
    "conditioning factors", "observation records", "satellite-based data",
)
DATASET_PATTERN = re.compile(
    r"\b(?:dataset|data set|database|corpus|benchmark|inventory|"
    r"conditioning factors|observation records|satellite[- ]based data|"
    r"[A-Z]{2,6}\s+(?:data|dataset))\b",
    re.IGNORECASE,
)
NAMED_DATASET_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&'’-]*(?:\s+[A-Z][A-Za-z0-9&'’-]*){0,5}\s+"
    r"(?:Dataset|Database|Corpus|Benchmark|Inventory))\b"
)

GEOGRAPHIC_SUFFIX_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,4}\s+"
    r"(?i:river|basin|watershed|catchment|lake|gulf|valley|delta|"
    r"province|state|county|region|island|mountain(?:s)?|city|town|"
    r"municipality))\b"
)
COUNTRY_AND_REGION_NAMES = (
    "United States", "United Kingdom", "South Korea", "North Korea",
    "Saudi Arabia", "New Zealand", "South Africa", "Iran", "China", "India",
    "Europe", "Africa", "Asia", "Canada", "Australia", "Japan", "Brazil",
    "Mexico", "Pakistan", "Bangladesh", "Turkey", "Türkiye", "Iraq", "Egypt",
    "Russia", "Germany", "France", "Italy", "Spain", "Indonesia", "Vietnam",
    "Thailand", "Nepal", "Afghanistan",
)
COUNTRY_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(name) for name in sorted(COUNTRY_AND_REGION_NAMES, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)
RELATED_WORK_CUES = re.compile(
    r"\b(?:previous|prior|earlier|other|related|existing)\s+"
    r"(?:stud(?:y|ies)|work|research|literature|authors?)\b"
    r"|\b(?:et\s+al\.?|cited|citation|according to)\b",
    re.IGNORECASE,
)
RELATED_SECTION_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s*)?(?:related work|literature review|"
    r"background|previous studies)\s*:?\s*$",
    re.IGNORECASE,
)
CURRENT_SECTION_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s*)?(?:study area(?:s)?|study region|"
    r"materials and methods|methods|methodology|experimental setup|"
    r"results|discussion|conclusion|case study)\s*:?\s*$",
    re.IGNORECASE,
)
CURRENT_STUDY_CUES = re.compile(
    r"\b(?:our study area|the study area|study areas?|study sites?|"
    r"the present study|this study|we (?:investigated|studied|analyzed|analysed)|"
    r"data (?:were )?collected from|observational (?:flood )?data from|"
    r"conducted in|carried out in|located in)\b",
    re.IGNORECASE,
)
TITLE_AUTHOR_PREFIX = re.compile(
    r"^\s*(?:title|authors?|author affiliation|correspondence)\s*:",
    re.IGNORECASE,
)
FINDING_ACTION_CUES = re.compile(
    r"\b(?:achiev(?:e|es|ed)|obtain(?:s|ed)|yield(?:s|ed)|"
    r"outperform(?:s|ed)|improv(?:e|es|ed)|increas(?:e|es|ed)|"
    r"reduc(?:e|es|ed)|decreas(?:e|es|ed)|higher|lower|greatest|"
    r"highest|best|top[- ]performing|superior|inferior|show(?:s|ed)|"
    r"report(?:s|ed)|"
    r"demonstrat(?:e|es|ed)|significantly (?:increased?|reduced?|improved?|"
    r"higher|lower|associated)|no significant (?:difference|effect))\b",
    re.IGNORECASE,
)
FINDING_OUTCOME_CUES = re.compile(
    r"\b(?:accuracy|AUC|Kappa|precision|recall|F1(?:[- ]score)?|"
    r"performance|error|sensitivity|specificity|susceptibility|"
    r"risk|outcome|effect|difference|association|rate|score|"
    r"model|algorithm|ensemble|prediction)\b",
    re.IGNORECASE,
)
DATA_COLLECTION_ONLY = re.compile(
    r"\b(?:data|records?|observations?|images?)\s+(?:were\s+)?"
    r"(?:collected|gathered|downloaded|obtained|compiled)\b",
    re.IGNORECASE,
)
INTENT_ONLY = re.compile(
    r"\b(?:aim(?:s|ed)?|designed|developed|proposed|used|using|"
    r"attempt(?:s|ed)?|to)\b[^.;]{0,40}\b(?:improv|increas|reduc|"
    r"decreas|achiev|predict|classif)\w*",
    re.IGNORECASE,
)
STUDY_PLACE_AFTER_CUE = re.compile(
    r"\b(?:was conducted|were conducted|carried out|performed|"
    r"investigated|located)\s+in\s+(?:the\s+)?"
    r"([A-Z][A-Za-z'’-]*(?:\s+[A-Z][A-Za-z'’-]*){0,3})"
)

METRIC_VALUE_PATTERN = re.compile(
    r"\b(?:overall accuracy|accuracy|kappa(?: coefficient)?|precision|recall|"
    r"F1(?:[- ]score)?|AUC|area under (?:the )?curve|RMSE|MAE|sensitivity|"
    r"specificity|balanced accuracy|IoU|MCC|R(?:2|²))\b"
    r"(?:\s*(?:was|of|=|:|>|<|>=|<=|≥|≤)\s*|\s*)"
    r"(?:[<>]=?|≥|≤)?\s*\d+(?:\.\d+)?\s*%?",
    re.IGNORECASE,
)

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def _matching_sentences(
    sentences: list[str],
    patterns: tuple[str, ...],
) -> list[str]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    return [
        sentence.strip()
        for sentence in sentences
        if any(pattern.search(sentence) for pattern in compiled)
    ]


def _unique_join(values: list[str]) -> str | None:
    unique = list(dict.fromkeys(value for value in values if value))
    return " | ".join(unique) if unique else None


def _extract_location(sentences: list[str]) -> str | None:
    candidates: list[str] = []
    related_section = False
    for sentence in sentences:
        if RELATED_SECTION_HEADING.match(sentence):
            related_section = True
            continue
        if CURRENT_SECTION_HEADING.match(sentence):
            related_section = False
            continue
        has_current_attribution = bool(CURRENT_STUDY_CUES.search(sentence))
        if has_current_attribution:
            related_section = False
        has_named_place = bool(
            GEOGRAPHIC_SUFFIX_PATTERN.search(sentence) or COUNTRY_PATTERN.search(sentence)
        )
        has_study_place = bool(STUDY_PLACE_AFTER_CUE.search(sentence))
        describes_dataset = bool(DATASET_PATTERN.search(sentence))
        if (
            (has_named_place or has_study_place)
            and not related_section
            and not RELATED_WORK_CUES.search(sentence)
            and not TITLE_AUTHOR_PREFIX.search(sentence)
            and not (describes_dataset and not has_named_place)
        ):
            candidates.append(sentence.strip())
    return _unique_join(candidates)


def _extract_dataset(sentences: list[str]) -> str | None:
    values = [
        sentence.strip()
        for sentence in sentences
        if DATASET_PATTERN.search(sentence) or NAMED_DATASET_PATTERN.search(sentence)
    ]
    return _unique_join(values)


def _extract_models(sentences: list[str]) -> str | None:
    return _unique_join(
        [
            sentence.strip()
            for sentence in sentences
            if MODEL_PATTERN.search(sentence)
        ]
    )


def _extract_ensemble_comparison(sentences: list[str]) -> str | None:
    pattern = re.compile(
        r"\b(?:ensemble|voting|stacking|bagging|boosting|individual model|"
        r"single model|base learner|best individual|best overall)\b",
        re.IGNORECASE,
    )
    performance_sentences = [
        sentence.strip()
        for sentence in sentences
        if pattern.search(sentence)
        or (
            MODEL_PATTERN.search(sentence)
            and METRIC_VALUE_PATTERN.search(sentence)
        )
    ]
    return _unique_join(performance_sentences)


def _extract_metric_values(sentences: list[str]) -> str | None:
    matches = [
        f"{match.group(0).strip()} — {sentence.strip()}"
        for sentence in sentences
        for match in METRIC_VALUE_PATTERN.finditer(sentence)
    ]
    return _unique_join(matches)


def _extract_findings(sentences: list[str]) -> str | None:
    findings: list[str] = []
    for sentence in sentences:
        if (
            DATA_COLLECTION_ONLY.search(sentence)
            or RELATED_WORK_CUES.search(sentence)
            or TITLE_AUTHOR_PREFIX.search(sentence)
            or INTENT_ONLY.search(sentence)
        ):
            continue
        has_metric_value = bool(METRIC_VALUE_PATTERN.search(sentence))
        has_result_action = bool(FINDING_ACTION_CUES.search(sentence))
        has_outcome = bool(FINDING_OUTCOME_CUES.search(sentence))
        has_named_model = bool(MODEL_PATTERN.search(sentence))
        if (
            (has_metric_value and (has_result_action or has_named_model))
            or (has_result_action and has_outcome)
        ):
            findings.append(sentence.strip())
    return _unique_join(findings)


def extract_context(text: str) -> dict[str, str | None]:
    """Extract context snippets; each value is verbatim source text from the passage.

    The caller retains the evidence chunk's paper name and page number. Therefore,
    every non-empty field can be traced to that exact source passage and page.
    """
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(text.strip())
        if sentence.strip()
    ]

    context: dict[str, str | None] = {
        "Dataset": _extract_dataset(sentences),
        "Study population": _unique_join(
            _matching_sentences(sentences, FIELD_PATTERNS["Study population"])
        ),
        "Geographic location": _extract_location(sentences),
        "Experimental conditions": _unique_join(
            _matching_sentences(sentences, FIELD_PATTERNS["Experimental conditions"])
        ),
        "Training and validation": _unique_join(
            _matching_sentences(sentences, FIELD_PATTERNS["Training and validation"])
        ),
        "Input variables": _unique_join(
            _matching_sentences(sentences, FIELD_PATTERNS["Input variables"])
        ),
        "Methodology": _unique_join(
            _matching_sentences(sentences, FIELD_PATTERNS["Methodology"])
        ),
        "Model or algorithm": _extract_models(sentences),
        "Evaluation metrics": _unique_join(
            _matching_sentences(sentences, FIELD_PATTERNS["Evaluation metrics"])
        ),
        "Metric values": _extract_metric_values(sentences),
        "Study period": _unique_join(
            _matching_sentences(sentences, FIELD_PATTERNS["Study period"])
        ),
        "Reported findings": _extract_findings(sentences),
        "Individual versus ensemble performance": _extract_ensemble_comparison(
            sentences
        ),
    }
    return context
