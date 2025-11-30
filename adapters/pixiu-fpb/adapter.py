"""
Adapter for PIXIU Financial PhraseBank (en-fpb) dataset to Terminal-Bench format.

The en-fpb dataset contains financial news sentences labeled with sentiment:
- positive
- negative  
- neutral
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm


@dataclass
class FPBRecord:
    """Represents a single FPB sample."""
    
    fpb_id: str
    sentence: str
    label: str
    dataset_name: str
    split: str
    
    @classmethod
    def from_hf_sample(cls, sample: Dict[str, Any], idx: int, dataset_name: str, split: str) -> "FPBRecord":
        """Create FPBRecord from HuggingFace dataset sample."""
        # The dataset has 'answer' field with the label string directly
        # and 'text' field with the sentence
        label = sample.get("answer", "neutral")
        sentence = sample.get("text", sample.get("sentence", ""))
        
        # Use the original ID if available, otherwise generate one
        fpb_id = sample.get("id", f"fpb{idx:06d}")
        
        return cls(
            fpb_id=fpb_id,
            sentence=sentence,
            label=label,
            dataset_name=dataset_name,
            split=split,
        )


class PixiuFPBAdapter:
    """Adapter to convert PIXIU FPB samples into Terminal-Bench tasks."""
    
    ALLOWED_CHOICES = ["negative", "neutral", "positive"]
    DIFFICULTY = "medium"
    
    def __init__(
        self,
        task_dir: Path,
        split: str = "test",
        dataset_name: str = "TheFinAI/en-fpb",
        limit: int | None = None,
    ):
        self.task_dir = Path(task_dir)
        self.split = split
        self.dataset_name = dataset_name
        self.limit = limit
        self.template_dir = Path(__file__).parent / "template"
        
    def load_dataset(self) -> List[FPBRecord]:
        """Load and convert HuggingFace dataset to FPBRecord list."""
        print(f"Loading {self.dataset_name} (split={self.split})...")
        try:
            ds = load_dataset(self.dataset_name, split=self.split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nNote: This is a gated dataset. Please:")
            print("1. Visit https://huggingface.co/datasets/TheFinAI/en-fpb")
            print("2. Request access and wait for approval")
            print("3. Run: huggingface-cli login")
            raise
        
        records = []
        # If limit is None or negative, use all samples
        limit = len(ds) if (self.limit is None or self.limit < 0) else self.limit
        for idx, sample in enumerate(ds):
            if idx >= limit:
                break
            records.append(FPBRecord.from_hf_sample(sample, idx, self.dataset_name, self.split))
        
        print(f"Loaded {len(records)} samples")
        return records
    
    def generate_all(self) -> List[str]:
        """Generate all Terminal-Bench tasks from the dataset."""
        records = self.load_dataset()
        self.task_dir.mkdir(parents=True, exist_ok=True)
        
        written_tasks = []
        for record in tqdm(records, desc="Generating FPB tasks"):
            task_id = f"pixiu-fpb-{record.fpb_id}"
            task_path = self.task_dir / task_id
            self._generate_task(record, task_path)
            written_tasks.append(task_id)
        
        return written_tasks
    
    def _generate_task(self, record: FPBRecord, out_dir: Path) -> None:
        """Generate a single Terminal-Bench task from an FPB record."""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy template files
        self._copy_template_files(out_dir)
        
        # Write data file
        self._write_item_file(out_dir, record)
        
        # Update task.yaml with specific values
        self._update_task_yaml(out_dir, record)
        
        # Update solution.sh with correct answer
        self._update_solution(out_dir, record)
        
        # Update test_outputs.py with expected values
        self._update_tests(out_dir, record)
    
    def _copy_template_files(self, out_dir: Path) -> None:
        """Copy template files to task directory."""
        for item in self.template_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, out_dir / item.name)
            elif item.is_dir() and item.name == "tests":
                shutil.copytree(item, out_dir / "tests", dirs_exist_ok=True)
    
    def _write_item_file(self, out_dir: Path, record: FPBRecord) -> None:
        """Write the FPB sample data to item.json inside tests/data/."""
        data_dir = out_dir / "tests" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "id": record.fpb_id,
            "label_type": "sentiment",
            "sentence": record.sentence,
            "choices": self.ALLOWED_CHOICES,
            "dataset": self.dataset_name,
            "split": self.split,
        }
        
        (data_dir / "item.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
    
    def _update_task_yaml(self, out_dir: Path, record: FPBRecord) -> None:
        """Update task.yaml with FPB-specific values."""
        task_yaml_path = out_dir / "task.yaml"
        content = task_yaml_path.read_text()
        
        choices_inline = ", ".join(self.ALLOWED_CHOICES)
        
        replacements = {
            "{pixiu_id}": record.fpb_id,
            "{label_type}": "sentiment",
            "{choices_inline}": choices_inline,
            "{difficulty}": self.DIFFICULTY,
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        task_yaml_path.write_text(content)
    
    def _update_solution(self, out_dir: Path, record: FPBRecord) -> None:
        """Update solution.sh with the correct answer."""
        solution_path = out_dir / "solution.sh"
        content = solution_path.read_text()
        content = content.replace("{expected_label}", record.label)
        solution_path.write_text(content)
    
    def _update_tests(self, out_dir: Path, record: FPBRecord) -> None:
        """Update test_outputs.py with expected values."""
        test_path = out_dir / "tests" / "test_outputs.py"
        content = test_path.read_text()
        
        replacements = {
            "{expected_label}": record.label,  # No extra quotes - template already has them
            "{allowed_choices}": json.dumps(self.ALLOWED_CHOICES),
            "{pixiu_id}": record.fpb_id,  # No extra quotes
            "{label_type}": "sentiment",  # No extra quotes
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        test_path.write_text(content)

