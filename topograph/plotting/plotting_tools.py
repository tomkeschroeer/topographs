def comp_pred_label(pred_f_val, label_f_val, slope, shift):
    cut_val = (slope * (1 - shift)) / 4
    pred_f_val = 1 if pred_f_val >= cut_val else 0
    return 1 if pred_f_val == label_f_val else 0


def comp_pred_label_pt(pred_f_val, label_f_val):
    return pred_f_val - label_f_val


class calculate_efficiency:
    def __init__(
        self, pred, label, Ntotal, slope, shift, zeros_only=False, ones_only=False
    ):
        self.pred_f = pred.flatten()
        self.label_f = label.flatten()
        self.Ntotal = Ntotal
        self.slope = slope
        self.shift = shift
        self.cut_val = (self.slope * (1 - self.shift)) / 4
        self.cut_val = 0.0623
        self.ones_only = ones_only
        self.zeros_only = zeros_only

    def __call__(self):
        if self.ones_only and not self.zeros_only:
            self.pred_f = self.pred_f[self.label_f == 1]
            Nsubset = sum(self.label_f == 1)
            n_true = list(map(self.comp_pred_label_ones_only, self.pred_f))
            return sum(n_true) / Nsubset
        if self.zeros_only and not self.ones_only:
            self.pred_f = self.pred_f[self.label_f == 0]
            Nsubset = sum(self.label_f == 0)
            n_true = list(map(self.comp_pred_label_zeros_only, self.pred_f))
            return sum(n_true) / Nsubset
        n_true = list(map(self.comp_pred_label, self.pred_f, self.label_f))
        return sum(n_true) / self.Ntotal

    def comp_pred_label(self, pred_f_val, label_f_val):
        pred_f_val = 1 if pred_f_val >= self.cut_val else 0
        return 1 if pred_f_val == label_f_val else 0

    def comp_pred_label_ones_only(self, pred_f_val):
        return 1 if pred_f_val >= self.cut_val else 0

    def comp_pred_label_zeros_only(self, pred_f_val):
        return 1 if pred_f_val <= self.cut_val else 0


def calculate_delta_pt(pred, label):
    pred_f = pred.flatten()
    label_f = label.flatten()
    reg = list(map(comp_pred_label_pt, pred_f, label_f))
    return reg
