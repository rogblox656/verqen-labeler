"""Label Studio exporter: import-ready task JSON (pure JSON).

Each result becomes one task with ``data.text`` and a ``predictions``
entry. The prediction's ``value.labels`` carries the certified conformal
set (a single label when coverage yields a set of one) and ``score`` is
the confidence. Import through the Label Studio UI (Import > Choose
Files), or with the SDK::

    from label_studio_sdk import Client

    ls = Client(url="http://localhost:8080", api_key="your-key")
    project = ls.get_project(project_id)
    project.import_tasks(json.load(open("labels_label_studio.json")))

The SDK is only needed for the upload step (``pip install
"verqen-labeler[label-studio]"``); writing the file is stdlib-only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["export_label_studio"]


def export_label_studio(results: Sequence[Any], path: str | Path) -> Path:
    """Write *results* as a Label Studio import file (JSON list of tasks)."""
    target = Path(path)
    tasks: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        tasks.append(
            {
                "id": index + 1,
                "data": {"text": result.text},
                "predictions": [
                    {
                        "model_version": "verqen",
                        "score": round(result.confidence, 4),
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                                "value": {
                                    "labels": list(result.conformal_set),
                                    "start": 0,
                                    "end": len(result.text),
                                },
                            }
                        ],
                    }
                ],
            }
        )
    target.write_text(json.dumps(tasks, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return target
