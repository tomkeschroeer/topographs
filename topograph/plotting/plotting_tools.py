from struct import calcsize
import numpy as np

def comp_pred_label(pred_f_val, label_f_val):
    pred_f_val = 1 if pred_f_val >= 0.4 else 0
    return 1 if pred_f_val == label_f_val else 0

def comp_pred_label_pt(pred_f_val, label_f_val):
    return pred_f_val - label_f_val

def calculate_efficiency(pred, label, Ntotal):
    pred_f = pred.flatten()
    label_f = label.flatten()
    print(list(pred_f[pred_f < 0.4]))
    n_true = list(map(comp_pred_label, pred_f, label_f))
    Ntotal = Ntotal
    return sum(n_true)/Ntotal

def calculate_jetwise_efficiency(pred, label):
    pred_f = [pred_tr.flatten() for pred_tr in pred]
    label_f = [label_tr.flatten() for label_tr in label]
    # effs = map()
    # print(pred_f)
    # print(label_f)

def calculate_delta_pt(pred, label):
    pred_f = pred.flatten()
    label_f = label.flatten()
    reg = list(map(comp_pred_label_pt, pred_f, label_f))
    return reg

 

