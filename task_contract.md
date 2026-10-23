# Task Contract: Structured Extraction from College Event Notices

Frozen specification for the PromptRepair study. This contract is written
before optimization and must not change after final-test results are seen.

## Task

Every input contains one clearly named target college event. The system must
return exactly one JSON object describing it with these fields:

| Field | Allowed value | Meaning |
|---|---|---|
| `event` | Nonempty string | The stated name of the target event |
| `date` | Nonempty string or JSON `null` | Its explicitly stated event date |
| `venue` | Nonempty string or JSON `null` | Its explicitly stated venue |

## Content rules

1. Copy the relevant wording from the notice verbatim. Do not reformat dates
   (preserve `20 October 2026`, not `2026-10-20`). Evaluation collapses repeated
   whitespace only; arbitrary paraphrases or date formats are not equivalent.
2. An absent date or venue becomes `null`.
3. A field with two unresolved conflicting values becomes `null`.
4. An explicit correction ("rescheduled from X to Y") selects Y; it is not an
   unresolved conflict.
5. A registration deadline, publication date, or previous event date is not
   the current event date.
6. Every key must be present. `null`, `"null"`, `"N/A"`, an empty string, and
   an omitted key are not interchangeable.

## Output rules

1. Return exactly one JSON object and no other text: no Markdown fences, no
   explanations, no surrounding prose.
2. The object must contain exactly the keys `event`, `date`, and `venue`.
   No additional keys.
3. `event` is a nonempty string. `date` and `venue` are each either a nonempty
   string or JSON `null`.

## Scope

- English, text-only, fictional academic notices.
- One clearly identifiable target event per notice.
- Three output fields only.
- The validation schema is a local checking specification; in the main
  experiment it is not supplied as a provider-enforced response schema.

## Reference example

Notice:

```text
The AI Club announces the Prompt Engineering Workshop.
The workshop will take place on 20 October 2026.
The venue will be announced later.
```

Reference answer:

```json
{"event": "Prompt Engineering Workshop", "date": "20 October 2026", "venue": null}
```

"The venue will be announced later" does not identify a venue and must not
become the venue string.
