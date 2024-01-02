from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

class MetricCalculator:
    def __init__(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])
        self.y_score = np.array([])
        
    def update(self, y_true, y_pred, y_score=np.array([])):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.y_score = np.concatenate([self.y_score, y_score])

    def calc_auc(self):
        return roc_auc_score(self.y_true, self.y_score)
    
    def calc_acc(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def clean(self):
        self.y_true = []
        self.y_pred = []
        self.y_score = []