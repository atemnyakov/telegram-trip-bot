import torch
import torch.nn.functional as F
from transformers import Trainer

def focal_loss(logits, labels, gamma=2.0, reduction='mean'):
    """
    Computes Focal Loss, which down-weights easy examples and focuses on hard ones.

    Args:
        logits: Tensor of shape [batch_size, num_classes] - raw model outputs.
        labels: Tensor of shape [batch_size] - ground truth class indices.
        gamma: Focusing parameter; higher values put more weight on hard examples.
        reduction: 'mean' (default) returns mean loss, 'sum' sums all losses.

    Returns:
        Loss value (scalar).
    """
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
    prob = torch.exp(-ce_loss)
    focal = (1 - prob) ** gamma * ce_loss

    return focal.mean() if reduction == 'mean' else focal.sum()

class RouteParserTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        loss = focal_loss(logits, labels, gamma=2.0)  # Using focal loss

        return (loss, outputs) if return_outputs else loss
