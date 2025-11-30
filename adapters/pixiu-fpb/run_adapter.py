import argparse
from pathlib import Path

from adapter import PixiuFPBAdapter

T_BENCH_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PIXIU FPB Terminal-Bench tasks")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=T_BENCH_ROOT / "tasks" / "pixiu-en-fpb",
        help="Directory where generated tasks will be stored.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="FPB split to convert.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of FPB samples to convert (for quick iterations).",
    )
    parser.add_argument(
        "--dataset-name",
        default="TheFinAI/en-fpb",
        help="Hugging Face dataset identifier to load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = PixiuFPBAdapter(
        task_dir=args.output_dir,
        split=args.split,
        dataset_name=args.dataset_name,
        limit=args.limit,
    )
    written = adapter.generate_all()
    print(f"Generated {len(written)} FPB tasks under {args.output_dir}")


if __name__ == "__main__":
    main()

