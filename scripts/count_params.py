"""Print parameter counts for every MODEL_CONFIGS entry and check the 50M cap.

Usage: uv run python scripts/count_params.py
"""
import json

from gibc.configs import MODEL_CONFIGS, PARAM_CAP
from gibc.model import count_params


def main():
    report = {}
    for name, cfg in MODEL_CONFIGS.items():
        c = count_params(cfg)
        # Trainable parameters: every tensor in model.parameters() counted once. The tied
        # embedding/output matrix is one tensor and is included.
        report[name] = {"trainable": c["once"], "embedding": c["embedding"],
                        "non_embedding": c["non_embedding"], "under_cap": c["once"] <= PARAM_CAP}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
