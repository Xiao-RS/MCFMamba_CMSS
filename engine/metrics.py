"""Evaluation Metrics for Semantic Segmentation"""

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from .utils import log_msg


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list = None, ignore_labels: int = -100):
        self.metric_df = None
        self.acc = 0
        self.metric_dict = {"Acc": 0, "Kappa": 0, "FWIoU": 0}
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.ignore_labels = ignore_labels

    def reset(self):
        self.metric_df = None
        self.matrix.fill(0)

    def update(self, preds, labels):
        assert (preds.dtype == "int64") and (np.ndim(preds) == 1), 'predicted values are not onehot or one dimension'
        assert preds.shape[0] == labels.shape[0], 'number of targets and predicted outputs do not match'
        assert (preds.max() < self.num_classes) and (preds.min() >= 0), 'predicted values are not between 0 and k-1'

        mask = (labels != self.ignore_labels)
        x = preds[mask] + self.num_classes * labels[mask]
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.num_classes ** 2)
        assert bincount_2d.size == self.num_classes ** 2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.matrix += conf
        self.acc = np.diag(self.matrix).sum() / self.matrix.sum()

    def statistics(self):
        self.metric_dict["Acc"] = po = np.diag(self.matrix).sum() / self.matrix.sum()
        pe = np.dot(self.matrix.sum(axis=0), self.matrix.sum(axis=1)) / (self.matrix.sum() ** 2)
        self.metric_dict["Kappa"] = (po - pe) / (1 - pe)
        self.metric_dict["FWIoU"] = 0
        class_metric = {}
        for i in range(self.num_classes):
            # columns: predicted targets. P\L TP FP
            # rows: ground-truth targets.     FN TN
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = (TP / (TP + FP)) if TP + FP != 0 else 0.
            Recall = (TP / (TP + FN)) if TP + FN != 0 else 0.
            Specificity = (TN / (TN + FP)) if TN + FP != 0 else 0.
            IoU = (TP / (TP + FN + FP)) if TP + FN + FP != 0 else 0.
            Dice = (2 * TP / (2 * TP + FN + FP)) if 2 * TP + FN + FP != 0 else 0.
            class_metric[self.labels[i]] = {
                "Precision": Precision, "Recall": Recall, "Specificity": Specificity,
                "IoU": IoU, "Dice": Dice
            }
            self.metric_dict["FWIoU"] += ((TP + FN) / (TP + FP + TN + FN)) * IoU
        self.metric_df = pd.DataFrame(class_metric).T
        self.metric_df.loc["mean"] = self.metric_df.mean()
