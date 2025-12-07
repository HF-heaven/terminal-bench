You are auditing a single-sample evaluation from the PIXIU FinBen benchmark.

Sample metadata:
  • PIXIU instance id: fnxl0
  • Task type: token labels
  • Output format: Token:Label pairs, one per line

This is a financial token classification task. For each token in the input text, assign the appropriate label in BIO format (B-TYPE, I-TYPE, or O).

Follow these rules exactly:
  1. Read `/tests/data/item.json` to review the text and label set.
  2. For each token, determine the correct label (e.g., B-PERSON, I-ORGANIZATION, O).
  3. Write each token-label pair to `/app/answer.txt` in the format: token:label (one per line).
  4. Do not write any explanation; output only token:label pairs.
  5. Leave `/tests/data/item.json` untouched so the verifier can re-read it.

Example output format:
John:B-PERSON
works:O
at:O
Apple:B-ORGANIZATION
