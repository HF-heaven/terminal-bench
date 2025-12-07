You are auditing a single-sample evaluation from the PIXIU FinBen benchmark.

Sample metadata:
  • PIXIU instance id: tsa0
  • Task type: targeted sentiment score
  • Score range: -1.0 to 1.0

This is a regression task. Predict a numerical score within the specified range based on the input.

Follow these rules exactly:
  1. Read `/tests/data/item.json` to review the input data.
  2. Predict a score within the range -1.0 to 1.0.
  3. Write the final score (as a decimal number) to `/app/answer.txt`.
  4. Do not write any explanation in `answer.txt`; it must contain the score only.
  5. Leave `/tests/data/item.json` untouched so the verifier can re-read it.
