BATCH 12F — FIRST FIVE PAIRS: SOURCE-CHECKED FIRST PASS

Files
- Batch12F_Pilot_Review_Workbook.xlsx: proposed annotations, source verification, review queue, and instructions.
- Batch12F_First5_Source_Checked_Annotations.csv: compact annotation table.
- Batch12F_Updated_Annotation_Queue.csv: original 30-pair queue with first five filled; remaining 25 unchanged.

Important status distinction
These are AI-assisted first-pass annotations checked against the original source PDF pages. They are NOT independent human annotations, NOT adjudicated ground truth, and NOT an inter-annotator agreement study. A human reviewer must accept, revise, or reject each proposed label. A second annotator must work independently before agreement statistics can be calculated.

First-pass result
All five pairs are proposed as METHODOLOGICAL_DIFFERENCE. The papers share the broad theme of evaluating automated factuality/hallucination evaluators against human judgments, but they use different datasets, metrics, units of analysis, and annotation/reference protocols. None is a confirmed contradiction.

Key source nuance
- P006 Table 3: VERIFY ties FactCheck-GPT on factual Pearson correlation (.97) and is higher on Other labels (.73 vs .66); avoid saying it wins every category.
- P007: GPT-4 has the highest average among individual metrics in the studied Begin/HaluEval setup (PRAUC .74 in Table 1; weighted-F1 .74 in Table 10).
- P010: VeriScore-human agreement is reported as kappa .27 / raw 69%, but the authors warn that annotators mainly used VeriScore web snippets as reference.
- P012: SAFE-human agreement is 72% over 16,011 facts; the 76/19/5 result comes from 100 initially disputed cases re-annotated with Google Search, not the full sample.

Next step
Review the first five proposed labels in the workbook. Do not calculate Cohen's kappa from AI-first-pass labels and one human review. For genuine IAA, obtain a second independent human annotation on the same five pairs, then adjudicate disagreements.
