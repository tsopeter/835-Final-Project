import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossWithMask(nn.Module):
    def __init__(self, margin : float = 0.2):
        super(TripletLossWithMask, self).__init__()
        self.margin = margin
        self.lossfn = torch.nn.TripletMarginLoss(margin=self.margin)

    def forward(self, features, mask):
        # features [bsz, features]
        # mask [bsz, bsz]

        if len(features.shape) > 2:
            features = features.unsqueeze(1)

        anchor = []
        positive = []
        negative = []

        bsz, dim = features.shape[0], features.shape[1]
        device = features.device

        diag_mask = torch.eye(bsz, dtype=torch.bool, device=device)
        neg_mask  = ~mask.bool() & ~diag_mask

        indicies  = torch.arange(bsz, device=device)

        for i in range(bsz):
            pos_i = indicies[mask[i] == 1]
            neg_i = indicies[neg_mask[i] == 1]

            if len(pos_i) == 0 or len(neg_i) == 0:
                continue

            pos_grid, neg_grid = torch.meshgrid(pos_i, neg_i, indexing='ij')
            pos_flat, neg_flat = pos_grid.reshape(-1), neg_grid.reshape(-1)

            anchor_embed = features[i].unsqueeze(0).expand(pos_flat.size(0), dim)
            pos_embed    = features[pos_flat]
            neg_embed    = features[neg_flat]

            anchor.append(anchor_embed)
            positive.append(pos_embed)
            negative.append(neg_embed)

        if len(anchor) == 0:
            return torch.tensor(0, device=device)
        
        anchor = torch.cat(anchor, dim=0)
        positive = torch.cat(positive, dim=0)
        negative = torch.cat(negative, dim=0)

        loss = self.lossfn(anchor, positive, negative)
        return loss

