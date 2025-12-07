You are auditing a single-sample evaluation from the PIXIU FinBen benchmark.

Sample metadata:
  • PIXIU instance id: tatqa0
  • Task type: text answer
  • Allowed labels: our allowable incurred costs plus a profit which can be fixed or variable depending on the contract’s fee arrangement up to predetermined funding levels determined by the customer

This is a table-based question answering task. Analyze the provided table and text, extract or infer the answer, and output it as text.

Follow these rules exactly:
  1. Read `/tests/data/item.json` to review the table, context, and question.
  2. Extract or infer the answer from the table and text.
  3. Write the final text answer to `/app/answer.txt`.
  4. Do not write any explanation in `answer.txt`; it must contain the answer only.
  5. Leave `/tests/data/item.json` untouched so the verifier can re-read it.
