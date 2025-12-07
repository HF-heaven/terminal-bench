You are auditing a single-sample evaluation from the PIXIU FinBen benchmark.

Sample metadata:
  • PIXIU instance id: finqa0
  • Task type: numerical answer
  • Allowed labels: 94.0

This is a financial numerical reasoning task. Analyze the provided financial data, perform necessary calculations, and output the correct numerical answer.

Follow these rules exactly:
  1. Read `/tests/data/item.json` to review the question and financial data.
  2. Perform any necessary calculations or reasoning to derive the answer.
  3. Write the final numerical answer (as a number only, no units or explanation) to `/app/answer.txt`.
  4. Do not write any explanation in `answer.txt`; it must contain the number only.
  5. Leave `/tests/data/item.json` untouched so the verifier can re-read it.
