import torch.nn as nn
import pytorch_tools as pt
from functools import partial
from pytorch_tools.utils.box import generate_targets


class DetectionLoss(nn.Module):
    """Constructs Wrapper for Object Detection which combines Focal Loss and SmoothL1 loss"""

    def __init__(
        self,
        anchors,
        focal_gamma=2,
        focal_alpha=0.25,
        huber_delta=0.1,
        box_weight=50,
        matched_iou=0.5,
        unmatched_iou=0.4,
    ):
        super().__init__()
        self.cls_criterion = pt.losses.FocalLoss(reduction="none", gamma=focal_gamma, alpha=focal_alpha)
        # TODO: compare to L1 loss. Mmdetection authors say it's better
        self.box_criterion = pt.losses.SmoothL1Loss(delta=huber_delta, reduction="none")
        self.generate_targets = partial(
            generate_targets, anchors=anchors, matched_iou=matched_iou, unmatched_iou=unmatched_iou
        )
        self.box_weight = box_weight

    def forward(self, outputs, target):
        """
        Args:
            outputs (Tuple[torch.Tensor]): cls_outputs, box_outputs
            target (torch.Tensor): shape [BS x N x 5]
        """
        cls_out, box_out = outputs
        cls_t, box_t, matches = self.generate_targets(batch_gt_boxes=target, num_classes=cls_out.size(2))
        # use foreground and background for classification and only foreground for regression
        num_fg = (matches > 0).sum() + 1
        # TODO: add fg EMA normaizer. https://github.com/open-mmlab/mmdetection/pull/2780
        # it is said to give additional ~0.5 AP
        cls_loss = self.cls_criterion(cls_out, cls_t)[matches >= 0].sum() / num_fg
        box_loss = self.box_criterion(box_out, box_t)[matches > 0].sum() / num_fg
        loss = cls_loss + self.box_weight * box_loss
        return loss
