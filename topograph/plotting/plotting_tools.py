from struct import calcsize
import numpy as np

pred = np.array([[[1],[0],[1],[1],[0],[0],[0],[1],[1],[0],[0]],[[1],[0],[1],[0],[0],[1],[0],[0],[1],[0],[0]]])
label = np.array([[[0],[1],[0],[1],[0],[0],[0],[1],[1],[0],[0]],[[1],[0],[0],[1],[0],[1],[0],[0],[1],[0],[0]]])

def comp_pred_label(pred_f_val, label_f_val):
    return 1 if pred_f_val == label_f_val else 0

def calculate_efficiency(pred, label):
    pred_f = pred.flatten()
    label_f = label.flatten()
    n_true = list(map(comp_pred_label, pred_f, label_f))
    return sum(n_true)/len(pred)

def calculate_jetwise_efficiency(pred, label):
    pred_f = [pred_tr.flatten() for pred_tr in pred]
    label_f = [label_tr.flatten() for label_tr in label]
    effs = map()
    print(pred_f)
    print(label_f)

calculate_jetwise_efficiency(pred, label)