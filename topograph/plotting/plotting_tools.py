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
        # self.cut_val = 0.0623
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

class calculate_binary_preds:
    def __init__(self, preds, slope, shift):
        self.preds = preds
        self.slope = slope
        self.shift = shift
        self.cut_val = (self.slope * (1 - self.shift)) / 4

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
