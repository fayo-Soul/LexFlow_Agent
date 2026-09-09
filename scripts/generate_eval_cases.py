#!/usr/bin/env python3
"""Generate the deterministic 30-case synthetic MVP evaluation set."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "eval" / "dev_set" / "sample_cases.json"
TARGETS = [
    SOURCE,
    ROOT / "lexflow_agent" / "data" / "eval" / "dev_set" / "sample_cases.json",
]


def main() -> None:
    seeds = json.loads(SOURCE.read_text(encoding="utf-8"))[:3]
    cases = []
    for index in range(30):
        seed = deepcopy(seeds[index % len(seeds)])
        number = index + 1
        seed["case_id"] = f"eval_{number:03d}"
        seed["metadata"] = {
            "synthetic": True,
            "split": "holdout" if number > 25 else "development",
            "variant": number,
        }
        for document in seed["documents"]:
            document["document_id"] = f"{document['document_id']}_{number:03d}"
            for page in document["pages"]:
                page["text"] = page["text"].replace("张三", f"测试员工{number:02d}")
                page["text"] = page["text"].replace("员工B", f"测试员工{number:02d}")
                page["text"] = page["text"].replace("公司A", f"测试企业{number:02d}")
        cases.append(seed)

    payload = json.dumps(cases, ensure_ascii=False, indent=2)
    for target in TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    print(f"generated={len(cases)}")


if __name__ == "__main__":
    main()
