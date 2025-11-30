"""
PIXIU → Terminal-Bench adapter.

This adapter converts PIXIU's FinBen classification samples into Terminal-Bench
tasks. Each task asks the agent to inspect one JSON file describing the sample
and to write the predicted label to /app/answer.txt.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from datasets import Dataset, load_dataset
from tqdm import tqdm

TEMPLATE_DIR = Path(__file__).parent / "template"


@dataclass(frozen=True)
class PixiuRecord:
    """Lightweight view over a PIXIU FinBen sample."""

    pixiu_id: str
    query: str
    choices: Sequence[str]
    gold_index: int
    label_type: str

    @property
    def expected_label(self) -> str:
        return self.choices[self.gold_index]


class PixiuAdapter:
    """Adapter that materialises PIXIU FinBen samples as Terminal-Bench tasks."""

    def __init__(
        self,
        task_dir: Path,
        split: str = "test",
        dataset_name: str = "TheFinAI/flare-headlines",
        limit: int | None = None,
    ) -> None:
        self.task_dir = Path(task_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dataset:
        ds = load_dataset(self.dataset_name, split=self.split)
        if self.limit is not None:
            max_rows = min(self.limit, len(ds))
            ds = ds.select(range(max_rows))
        return ds

    def iter_records(self) -> Iterable[PixiuRecord]:
        for row in self.dataset:
            yield PixiuRecord(
                pixiu_id=row["id"],
                query=row["query"],
                choices=row["choices"],
                gold_index=int(row["gold"]),
                label_type=row.get("label_type", "financial classification"),
            )

    def generate_task(self, record: PixiuRecord) -> None:
        task_id = self._tbench_id(record.pixiu_id)
        out_dir = self.task_dir / task_id
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(TEMPLATE_DIR, out_dir)

        self._write_item_file(out_dir, record)
        self._update_task_yaml(out_dir, record)
        self._update_solution(out_dir, record)
        self._update_tests(out_dir, record)

    def generate_all(self) -> List[str]:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        written_ids: List[str] = []
        for record in tqdm(self.iter_records(), desc="Generating PIXIU tasks"):
            self.generate_task(record)
            written_ids.append(self._tbench_id(record.pixiu_id))
        return written_ids

    def _write_item_file(self, out_dir: Path, record: PixiuRecord) -> None:
        # Put data inside tests/ directory so it gets copied to the container
        data_dir = out_dir / "tests" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": record.pixiu_id,
            "label_type": record.label_type,
            "query": record.query,
            "choices": list(record.choices),
            "dataset": self.dataset_name,
            "split": self.split,
        }
        (data_dir / "item.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _update_task_yaml(self, out_dir: Path, record: PixiuRecord) -> None:
        path = out_dir / "task.yaml"
        content = path.read_text()
        replacements = {
            "{pixiu_id}": record.pixiu_id,
            "{label_type}": record.label_type,
            "{choices_inline}": ", ".join(record.choices),
            "{difficulty}": "medium",
        }
        for key, value in replacements.items():
            content = content.replace(key, value)
        path.write_text(content)

    def _update_solution(self, out_dir: Path, record: PixiuRecord) -> None:
        path = out_dir / "solution.sh"
        path.write_text(
            path.read_text().replace("{expected_label}", record.expected_label)
        )

    def _update_tests(self, out_dir: Path, record: PixiuRecord) -> None:
        path = out_dir / "tests" / "test_outputs.py"
        replacements = {
            "{expected_label}": record.expected_label,
            "{allowed_choices}": json.dumps(list(record.choices), ensure_ascii=False),
            "{pixiu_id}": record.pixiu_id,
            "{label_type}": record.label_type,
        }
        content = path.read_text()
        for key, value in replacements.items():
            content = content.replace(key, value)
        path.write_text(content)

    @staticmethod
    def _tbench_id(pixiu_id: str) -> str:
        return f"pixiu-headlines-{pixiu_id.lower()}"


