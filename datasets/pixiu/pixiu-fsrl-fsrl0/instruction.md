You are auditing a single-sample evaluation from the PIXIU FinBen benchmark.

Sample metadata:
  • PIXIU instance id: fsrl0
  • Task type: semantic role labels
  • Output format: Token:Role pairs, one per line

This is a financial semantic role labeling task. For each token, assign the appropriate semantic role (e.g., Agent, Patient, Predicate).

Follow these rules exactly:
  1. Read `/tests/data/item.json` to review the sentence and role definitions.
  2. For each token, determine its semantic role in the financial context.
  3. Write each token-role pair to `/app/answer.txt` in the format: token:role (one per line).
  4. Do not write any explanation; output only token:role pairs.
  5. Leave `/tests/data/item.json` untouched so the verifier can re-read it.
