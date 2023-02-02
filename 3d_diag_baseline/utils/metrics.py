import numpy as np
import torch
from ignite.metrics import Metric
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)


class SaveOutputs(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self.outputs = []
        self.targets = []
        super(SaveOutputs, self).__init__(output_transform=output_transform, device=device)
    def reset(self):
        super(SaveOutputs, self).reset()
    def update(self, output):
        #self.outputs.append(output[0].cpu().detach().numpy())
        self.outputs.append(float(torch.sigmoid(output[0]).cpu().detach().numpy()[0,1]))
        self.targets.append(float(output[1].cpu().detach()))
    def compute(self):
        return np.array(self.outputs), np.array(self.targets)


def get_auroc(outputs, targets):
    auroc=roc_auc_score(targets, outputs)
    return np.round(auroc, 3)


def get_auprc(outputs,targets):
    precision, recall, _ = precision_recall_curve(targets, outputs)
    auprc = auc(recall, precision)
    return np.round(auprc, 3)


def get_acc(outputs, targets, threshold=0.5):
    preds=np.zeros(outputs.shape)
    preds[outputs>=threshold]=1
    preds[outputs<threshold]=0
    acc=accuracy_score(targets,preds)
    return np.round(acc, 3)


def get_f1_score(outputs, targets, threshold=0.5):
    preds=np.zeros(outputs.shape)
    preds[outputs>=threshold]=1
    preds[outputs<threshold]=0
    f1=f1_score(targets,preds)
    return np.round(f1, 3)


def get_sensitivity_specificity_precsion_per_class(outputs, targets, threshold=0.5, labels=[0, 1]):
    preds=np.zeros(outputs.shape)
    preds[outputs>=threshold]=1
    preds[outputs<threshold]=0
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=labels).ravel()
    Sensitivity = tp / (tp+fn+1e-8)
    Specificity = tn / (tn+fp+1e-8)
    Precision = tp / (tp+fp+1e-8)   
    return np.round(Sensitivity, 3), np.round(Specificity, 3), np.round(Precision, 3)


def get_performances(outputs, targets):
    ACC, f1_score, sensitivity, specificity, precision = None, None, None, None, None
    auroc = get_auroc(outputs, targets)
    auprc = get_auprc(outputs, targets)
    threshold_range = [0, 1]
    graduation = (threshold_range[1]-threshold_range[0])/1000
    threshold_list=list(np.round(np.linspace(threshold_range[0]+graduation,threshold_range[1]-graduation,999),3))
    min_abs_ss_sp = 1
    for threshold in threshold_list:
        acc = get_acc(outputs, targets, threshold=threshold)
        f1 = get_f1_score(outputs, targets, threshold=threshold)
        ss, sp, pr = get_sensitivity_specificity_precsion_per_class(outputs, targets, threshold=threshold)
        if np.abs(ss-sp)<=min_abs_ss_sp:
            min_abs_ss_sp = np.abs(ss-sp)
            ACC = acc
            f1_score = f1
            sensitivity = ss
            specificity = sp
            precision = pr
    return auroc, auprc, ACC, f1_score, sensitivity, specificity, precision
