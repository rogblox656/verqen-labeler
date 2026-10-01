"""Active learning loop: score an unlabeled pool, review the informative samples.

Requires a local checkpoint::

    export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
    python examples/active_learning_loop.py

The loop scores every pool sample with one cheap engine decision, ranks the
pool by uncertainty (entropy, margin, conformal set size), sends the top-k
most uncertain samples to human review and auto-labels the rest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow running without pip install

from verqen_labeler import Labeler  # noqa: E402
from verqen_labeler.active_learning import pick_top_k, uncertainty_score  # noqa: E402
from verqen_labeler.engine import load_engine  # noqa: E402

CATEGORIES = ["sports", "politics", "technology", "entertainment", "finance"]
QUESTION = "What is the correct label for this text?"

POOL = [
    "The striker scored twice in the second half to seal the win.",
    "Lawmakers debated the new immigration bill late into the evening.",
    "The new phone chip doubles battery life on a single charge.",
    "The sequel earned record box office receipts on opening weekend.",
    "The central bank raised interest rates again this quarter.",
]


def main() -> None:
    engine = load_engine()  # reads VERQEN_CHECKPOINT
    labeler = Labeler(categories=CATEGORIES, engine=engine)

    # 1. Score the whole pool (each decision carries probabilities and set).
    decisions = [engine.decide(f"Text: {text}", QUESTION, CATEGORIES) for text in POOL]

    # 2. The k most uncertain samples go to human review.
    review_indices = set(pick_top_k(decisions, k=2))

    # 3. Everything else is auto-labeled with the certified labeler.
    for index, (text, decision) in enumerate(zip(POOL, decisions)):
        score = uncertainty_score(decision)
        if index in review_indices:
            print(f"human   {score:.3f}  {text}")
        else:
            result = labeler.label(text)
            print(f"auto    {score:.3f}  {result.label:15} {text}")


if __name__ == "__main__":
    main()
