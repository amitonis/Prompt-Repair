# Step 1 blind judgments — provenance

**Status:** Step 1 submitted before opening `audit_key_COORDINATOR_ONLY.json` or `data/test_labels.jsonl` for this audit. No model output or `runs/ERROR_ANALYSIS.md` was consulted during this Step 1 pass.

## Coordinator-designated snapshot

The coordinator designated the files by their hashes and cited `analysis_plan.md` Section 10, run manifests, and coordinator records. The coordinator message left the contract hash as `<hash>`; Section 10 supplies the exact value. Start-of-Step-1 hashes:

- `task_contract.md`: `ca2003cd1e34ffd33d9313de2c0c7d056581b21b08d5cea702f972c756794622`
- `data/test_inputs.jsonl`: `39c91a5f2fd532193cecf69772c310a6996b561f32fa6be217e28e0e30a95e78`
- `data/test_labels.jsonl`: `6cff144787e9207ae7743f610793d77acf7e7ea7c0dc287489d7cf83e8a92248`

The three values match the Section 10 hash block in `analysis_plan.md` and the present files. The frozen contract includes A2's full-venue-span sentence.

## Limitation on blindness

Before this audit was initiated, the assistant had prior project context and had read the primary test labels and `runs/ERROR_ANALYSIS.md` during earlier work. For this Step 1 pass it used the coordinator-provided blind sheet and the designated contract, and did not reopen labels, model outputs, or the error analysis. Nevertheless, prior exposure means this is **not a fully independent blind annotation**. The judgments are submitted for protocol continuity, but the punctuation decisions should be independently replicated by a reviewer with no prior exposure before being presented as an independent audit result.

## Repository provenance

There is no `prereg-v1` tag. Git contains one `Initial files` commit; the test input and label files in that commit are empty. The contract blob in that commit hashes to `8b64c2d295524c46ad76826541f93eaf09ce8fa1bb499ca16030ebfea4c71b1f`, while the coordinator-designated frozen contract hashes to `ca2003cd...`. The working contract differs from the initial commit by the A2 sentence specifying that the venue is the full location phrase. The coordinator designated the Section 10 snapshot as authoritative despite the absence of Git evidence for the freeze; this limitation must be noted in the final report.
