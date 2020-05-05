from .keypoint_weighting import KeypointWeightingLoss
from .proposal import ProposalLoss

import torch.nn as nn

class OverallLoss(nn.Module):
    """
    TODO - finish all of this...
    """

    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.cfg = cfg
        self.proposal_loss = ProposalLoss()
        self.keypoint_weighting_loss = KeypointWeightingLoss()
        self.refinement_loss = RefinementLoss()

    def forward(self, item):
        keys = ['G_cls', 'M_cls', 'P_cls', 'G_reg', 'M_reg', 'P_reg']
        G_cls, M_cls, P_cls, G_reg, M_reg, P_reg = map(item.get, keys)
        proposal_loss = self.proposal_loss()
        keypoint_weighting_loss = self.keypoint_weighting_loss()
        refinement_loss = self.refinement_loss()
        losses = dict(proposal_loss=cls_loss, 
                      keypoint_weighting_loss=keypoint_weighting_loss, 
                      refinement_loss=refinement_loss)
        return losses
