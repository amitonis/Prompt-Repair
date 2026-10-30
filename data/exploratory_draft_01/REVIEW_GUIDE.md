# Exploratory draft 01 — reviewer label check

Status: **DRAFT — DO NOT RUN OR REPORT AS EVALUATION DATA UNTIL REVIEWED.**

This separate 12-case set was drafted to explore two error patterns from the
completed study: separating event titles from organizer names, and returning
`null` when sources give unresolved conflicting values. It does not alter the
frozen 60-case dataset, preregistered analysis, or reported final-test scores.

## Files

- `inputs.jsonl` — fictional notices, marked `exploratory_review_pending`.
- `labels_draft.jsonl` — proposed labels and evidence for review.

## Review decisions needed

1. For all six `EXP-EB` cases, confirm the event-title span and exclude the
   organizer when it is separate from the title.
2. For `EXP-EB-005`, decide whether the terminal exclamation mark is part of
   the event title under the verbatim-copy contract. The current label draft
   excludes it so this case directly tests the punctuation convention.
3. For the six `EXP-UC` cases, confirm that conflicting sources have no
   priority or correction and that the corresponding field should be `null`.
4. Verify every non-null label is copied exactly from the notice and that the
   event, date, and venue fields satisfy the task contract.

After review, mark each decision as accepted or corrected in a separate
review log. Keep the draft version unchanged for auditability. Only a reviewed
version should be used for an exploratory run, and its results must remain
separate from the primary study.
