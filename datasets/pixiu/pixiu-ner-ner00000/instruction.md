You are auditing a single-sample evaluation from the PIXIU FinBen benchmark.

Sample metadata:
  • PIXIU instance id: ner00000
  • Task type: named entity recognition (NER)
  • Output format: entity_name, entity_type (one per line)

This is a named entity recognition task. Identify all entities in the text and output them with their types.

Follow these rules exactly:
  1. Read `/tests/data/item.json` to review the text and entity types.
  2. Identify all named entities (persons, organizations, locations, etc.).
  3. Write each entity to `/app/answer.txt` in the format: entity_name, entity_type (one per line).
  4. Do not write any explanation; output only entity pairs.
  5. Leave `/tests/data/item.json` untouched so the verifier can re-read it.

Example output:
John Smith, PERSON
Apple Inc, ORGANIZATION
New York, LOCATION
