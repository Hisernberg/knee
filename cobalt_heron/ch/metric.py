"""Exact replica of the organizer's Self_Evaluation_Notebook PQ (V6 semantics).

Every (gt, pred) cell with IoU > 0.5 is a TP; FP = pred columns with no hit;
FN = gt rows with no hit.  Counts are pooled over all annotator records.
"""
import numpy as np
from pycocotools import mask as mu


class PQAccumulator:
    def __init__(self):
        self.s = 0.0; self.tp = 0; self.fp = 0; self.fn = 0

    def add(self, gt_rles, pred_rles):
        ng, npd = len(gt_rles), len(pred_rles)
        if ng == 0:
            self.fp += npd; return
        if npd == 0:
            self.fn += ng; return
        iou = np.asarray(mu.iou(list(pred_rles), list(gt_rles), [0] * ng)).T  # (ng, npd)
        hit = iou > 0.5
        self.s += float(iou[hit].sum()); self.tp += int(hit.sum())
        self.fp += int((hit.sum(0) == 0).sum()); self.fn += int((hit.sum(1) == 0).sum())

    def merge(self, o):
        self.s += o.s; self.tp += o.tp; self.fp += o.fp; self.fn += o.fn

    @property
    def pq(self):
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.s / d if d else 0.0

    def __repr__(self):
        sq = self.s / self.tp if self.tp else 0
        return f"PQ={self.pq:.4f} SQ={sq:.3f} TP={self.tp} FP={self.fp} FN={self.fn}"
