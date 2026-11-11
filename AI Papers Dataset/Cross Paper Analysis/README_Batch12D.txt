BATCH 12D PACKAGE
=================

Files
-----
1. Batch12D_Methodology_Specification.docx
   Formal research workflow, relation taxonomy, comparability gate, confidence rubric,
   uncertainty dimensions, evaluation design, guardrails and implementation sequence.

2. Batch12D_Relation_Annotation_Template.csv
   Empty row-level schema for annotating directed claim-to-claim evidence relationships.

3. Batch12D_Relation_and_Evaluation_Rubric.xlsx
   Relation labels, ordinal confidence rules, uncertainty dimensions, and evaluation plan.

4. Batch12D_Evaluation_Plan.csv
   Machine-readable evaluation components and caveats.

5. Batch12D_System_Architecture.mmd
   Mermaid architecture diagram source.

Important methodological limits
-------------------------------
- Batch 12C's current graph is an evidence-linkage graph, not a fully adjudicated
  support/contradiction graph. Do not relabel its edges as support or contradiction
  without applying the Batch 12D protocol.
- Confidence is ordinal (HIGH/MODERATE/LOW/NOT_ASSESSABLE), not a probability.
- Do not invent weighted confidence scores. Probability calibration requires a gold
  relation dataset and paper-level held-out evaluation.
- Keep unknown context fields as UNKNOWN.
- The current 21-paper corpus is a feasibility/pilot corpus. Avoid broad claims
  about all AI research.
- Preserve claims.csv as the canonical claim source; Batch 12D creates a new
  relation-analysis layer and does not overwrite the claim table.

Recommended next action
----------------------
Create 30-40 candidate claim pairs across the six existing research questions.
Have two annotators independently apply the template and rubric, then adjudicate
disagreements before scaling the relation graph.
