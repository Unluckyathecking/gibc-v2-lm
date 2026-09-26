"""Print parameter counts for every MODEL_CONFIGS entry and check the 50M cap.

Usage: uv run python scripts/count_params.py
"""
import json

from gibc.configs import MODEL_CONFIGS, PARAM_CAP
from gibc.model import count_params


def main():
    report = {}
    for name, cfg in MODEL_CONFIGS.items():
        counts = count_params(cfg)
        counts["under_cap_once"] = counts["once"] <= PARAM_CAP
        counts["under_cap_twice"] = counts["twice"] <= PARAM_CAP
        report[name] = counts
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
