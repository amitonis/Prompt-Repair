# Exploratory draft 02 — human-reviewed extension

Status: **HUMAN REVIEWED FOR POST-HOC EXPLORATORY USE**. Owner decisions and
authorization are recorded in `human_review.json`.

This v2 derives from draft 01. Draft 01 is preserved unchanged. The proposed
labels are review-only and are separate from the frozen 60-case dataset and
primary study results.

## Changes from draft 01

- EXP-EB-004 now uses a distinct event, organizer, and venue to avoid reusing
  the frozen SEL-006 entities. Its event and venue gold/evidence were updated
  to match the revised notice.
- EXP-UC-003 no longer calls one source "current"; its gold label remains
  `venue: null` because neither listed source has priority.
- All other notices and gold labels carry over unchanged.

## Recorded review decisions

1. The event title/organizer labels are accepted; EB-001 is recognized as low
   diagnostic value because it labels the title and organizer explicitly.
2. Quotation marks in EB-004 are treated as delimiters. This convention is
   recorded separately because contract text does not define it.
3. EB-005 excludes terminal `!` to match the A1 convention. Report EB-005
   separately from the other event-boundary cases.
4. The owner selected the EB-004 notice and matching event/venue label update
   to avoid frozen SEL-006 overlap.
5. UC-003 removes `current`; its venue gold remains null.
6. The remaining conflict labels are accepted under the stated no-priority
   wording.

## Exploratory reporting limits

There are 12 cases. Report counts only, mark all results **post hoc and
exploratory**, and keep them separate from the primary results. Five of six
conflict cases concern dates and one concerns venue. Every conflict notice
explicitly says neither source has priority, so these examples are easier than
real-world conflicts. Do not claim general performance from this small set.

The advisory Claude pass is recorded separately under draft 01. The owner
reviewed the decisions in chat and authorized an exploratory run. Keep those
results separate from the primary study.
