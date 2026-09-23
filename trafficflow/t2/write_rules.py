"""Write the rule-based candidate files (validation + private).

    PYTHONPATH=/home/user/knee python -m trafficflow.t2.write_rules
"""
from __future__ import annotations

import json

from .baselines import predict_rules
from .core import WORK
from .submit import check, write


def main():
    pv = predict_rules("validation"); pp = predict_rules("private")
    for m in pv:
        name = m.replace("+", "_")
        write({**pv[m], **pp[m]}, WORK / f"{name}.csv")
        print(m, json.dumps(check(WORK / f"{name}.csv")))


if __name__ == "__main__":
    main()
