"""Label a batch and export it as a Label Studio import file.

Requires a local checkpoint::

    export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
    python examples/label_studio_integration.py

Then import the generated ``label_studio_import.json`` in Label Studio
(Settings > Import > Choose Files), or push it with the SDK::

    from label_studio_sdk import Client

    ls = Client(url="http://localhost:8080", api_key="your-key")
    ls.get_project(PROJECT_ID).import_tasks(json.load(open("label_studio_import.json")))
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow running without pip install

from verqen_labeler import Labeler  # noqa: E402
from verqen_labeler.exporters import export  # noqa: E402

CATEGORIES = ["sports", "politics", "technology", "entertainment", "finance"]

TEXTS = [
    "The striker scored twice in the second half to seal the win.",
    "Lawmakers debated the new immigration bill late into the evening.",
    "The new phone chip doubles battery life on a single charge.",
    "The sequel earned record box office receipts on opening weekend.",
    "The central bank raised interest rates again this quarter.",
]

OUTPUT = Path("label_studio_import.json")


def main() -> None:
    labeler = Labeler(categories=CATEGORIES)  # reads VERQEN_CHECKPOINT
    results, report = labeler.label_dataset(TEXTS)

    path = export(results, OUTPUT, "label_studio")
    print(f"wrote {path} ({report.total} tasks, {report.human_review_needed} flagged for review)")
    print("Each task carries data.text plus a prediction with the certified label set.")
    print("Import in Label Studio: Settings > Import > Choose Files.")


if __name__ == "__main__":
    main()
