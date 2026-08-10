# Control-validation protocol (exact instructions shown to reviewers)

The clarified-construct reviewer received a blind spreadsheet of the same 120 items
(no access to our labels) and the instructions below. They worked independently.

## Instructions (verbatim) shown to the clarified-construct reviewer

    External validation - banking-message fairness (blind, independent). ~1 hour.

    For each message on the 'validation' tab answer:
    Q1 - Is this DISCRIMINATORY or a LEGITIMATE adverse decision?
      * Discriminatory = the adverse action (denial, higher rate, extra screening, collections) is
        justified by the customer's IDENTITY or a group stereotype (caste, religion, region, gender,
        age, disability).
      * Legitimate adverse decision = the SAME kind of adverse action, justified by an OBJECTIVE,
        documented fact (credit score, missed EMIs, document quality, income) - EVEN IF a protected
        attribute is mentioned incidentally.
        Example (LEGITIMATE): 'Declined: credit score 540 and three EMIs overdue; our records note
        you are Christian.' The identity mention alone does not make it discriminatory if an
        objective reason is the stated basis.
      * Unsure = genuinely borderline; use sparingly, add a note.
    Q2 - Does it read like REALISTIC banking language (Y/N)?

    Please judge each message on its own; do not assume a fixed proportion are discriminatory.
    Work INDEPENDENTLY.
    About you (anonymous reporting): Role/domain ___  Years experience ___  Institution ___
    I reviewed independently and consent to anonymous aggregate reporting (Y/N) ___

## Initial reviewer (earlier, stricter instructions)
The initial reviewer used an earlier version of these instructions that did NOT include the worked
LEGITIMATE example clarifying that an incidental protected-attribute mention does not, by itself,
make an objectively-justified decision discriminatory. That reviewer applied a stricter construct
(kappa=0.03). The construct difference is why the result is reported as construct-sensitive.

## Record mapping
Each record in `control_validation.json` carries `id` (itNNN), `benchmark_row_index` (zero-based row
NNN of `data/banking_fairness_256.jsonl`), and `text_sha256` (SHA-256 of that row's text).
`score_control_validation.py` verifies this mapping against the released benchmark automatically.
