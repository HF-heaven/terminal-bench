You are auditing a single-sample evaluation from the PIXIU FinBen benchmark.

Sample metadata:
  • PIXIU instance id: ectsum0
  • Task type: extractive summary labels
  • Output format: Binary sequence (0s and 1s)

This is an extractive summarization task. For each sentence in the document, output 1 if it should be included in the summary, 0 otherwise.

Follow these rules exactly:
  1. Read `/tests/data/item.json` to review the document.
  2. Determine which sentences are most important for the summary.
  3. Write a sequence of 0s and 1s to `/app/answer.txt` (one digit per sentence, space-separated or one per line).
  4. Do not write any explanation; output only the binary sequence.
  5. Leave `/tests/data/item.json` untouched so the verifier can re-read it.

Example output (for a 5-sentence document):
1 0 1 1 0
