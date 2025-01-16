# loss.py

import torch
import monai
from monai.losses import DiceCELoss, DiceLoss
import torch.nn.functional as F

LossFuncion = monai.losses.DiceLoss(include_background=True, to_onehot_y=False, softmax=True)
MetricDice = monai.metrics.DiceMetric(include_background=True, reduction="mean")


def BinaryOutput(output, keepdim=True):
    shape = output.shape
    argmax_idx = torch.argmax(output, axis=1, keepdim=True)
    argmax_oh = F.one_hot(argmax_idx, num_classes=4)
    if keepdim:
        argmax_oh = torch.squeeze(argmax_oh, dim=1)
    if len(shape) == 5:
        argmax_oh = argmax_oh.permute(0,4,1,2,3)
    elif len(shape) == 4:
        argmax_oh = argmax_oh.permute(0,3,1,2)

    return argmax_oh
