import numpy as np


def get_cut_val():
    cut_val = 0.6
    return cut_val


class calculate_efficiency:
    def __init__(
        self,
        pred,
        label,
        Ntotal,
        slope,
        shift,
        c1,
        c2,
        zeros_only=False,
        ones_only=False,
        cut_val=None,
    ):
        self.pred_f = pred.flatten()
        self.label_f = label.flatten()
        self.Ntotal = Ntotal
        self.cut_val = get_cut_val() if cut_val is None else cut_val
        self.ones_only = ones_only
        self.zeros_only = zeros_only

    def __call__(self):
        if self.ones_only and not self.zeros_only:
            one_true = self.label_f == 1
            self.pred_f = self.pred_f[one_true]
            Nsubset = sum(one_true)
            n_true = sum(self.pred_f > self.cut_val)
            return n_true / Nsubset
        if self.zeros_only and not self.ones_only:
            zero_true = self.label_f == 0
            self.pred_f = self.pred_f[zero_true]
            Nsubset = sum(zero_true)
            n_true = sum(self.pred_f <= self.cut_val)
            return n_true / Nsubset
        one_true = self.label_f == 1
        zero_true = self.label_f == 0
        n_true_one = sum(self.pred_f[one_true] > self.cut_val)
        n_true_zero = sum(self.pred_f[zero_true] <= self.cut_val)
        return (n_true_one + n_true_zero) / self.Ntotal

    def comp_pred_label(self, pred_f_val, label_f_val):
        pred_f_val = 1 if pred_f_val >= self.cut_val else 0
        return 1 if pred_f_val == label_f_val else 0

    def comp_pred_label_ones_only(self, pred_f_val):
        return 1 if pred_f_val >= self.cut_val else 0

    def comp_pred_label_zeros_only(self, pred_f_val):
        return 1 if pred_f_val <= self.cut_val else 0


class calculate_binary_preds:
    def __init__(self, preds):
        self.preds = preds
        self.cut_val = get_cut_val()

    def __call__(self):
        preds_bin = list(map(self.convert_preds_to_bin, self.preds))
        return preds_bin

    def convert_preds_to_bin(self, pred):
        pred = 1 if pred >= self.cut_val else 0
        return pred


class calculate_pT_diff:
    def __init__(self, pred, label):
        self.pred = pred
        self.label = label

    def __call__(self):
        pred_f = self.pred.flatten()
        label_f = self.label.flatten()
        reg = list(map(self.comp_pred_label_pt, pred_f, label_f))
        return reg

    def comp_pred_label_pt(self, pred_f_val, label_f_val):
        return pred_f_val - label_f_val
