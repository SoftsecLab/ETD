import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ==========================================
# Metrics Utilities
# ==========================================

def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


# ==========================================
# ETD Loss & Schedulers
# ==========================================

def calculate_ETD_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob,
                       beta=0.5, gama=0.5):

    prefered_relative_logprob = model_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob

    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob - gama)).mean(dim=-1)
    
    return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins


class LinearSchedule:
    """
    线性比例调度器，用于动态调整混合比例
    """

    def __init__(self, initial_value, final_value, total_steps):
        self.initial = initial_value
        self.final = final_value
        self.total_steps = total_steps
        self.delta = (final_value - initial_value) / total_steps
        self.current = initial_value

    def get_value(self, step):
        ratio = min(step / self.total_steps, 1.0)
        current_value = self.initial + ratio * (self.final - self.initial)
        return max(current_value, min(self.initial, self.final))  # 确保不越界