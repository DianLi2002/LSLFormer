import torch
import torch.nn as nn
import torch.nn.functional as F

class CMSCLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(CMSCLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, hsi_feats, lidar_feats):
        """
        hsi_feats: H2M output, shape [B, C_h, H, W]
        idar_feats: MCIE output, shape [B, C_l, H, W]
        """
        B, C_h, H, W = hsi_feats.shape
        N = H * W

        # Flatten spatial dimensions: [B, C, N]
        hsi_flat = hsi_feats.view(B, C_h, N)
        lidar_flat = lidar_feats.view(B, -1, N)

        # 1. Construct LiDAR Affinity Matrix (Target) ---
        # Normalize features for Cosine Similarity
        lidar_norm = F.normalize(lidar_flat, p=2, dim=1)
        # Compute similarity map: [B, N, N]
        sim_lidar = torch.bmm(lidar_norm.transpose(1, 2), lidar_norm)
        # Apply Softmax to get probability distribution P
        P_lidar = F.softmax(sim_lidar / self.temperature, dim=-1)

        # 2. Construct HSI Affinity Matrix (Input) ---
        hsi_norm = F.normalize(hsi_flat, p=2, dim=1)
        sim_hsi = torch.bmm(hsi_norm.transpose(1, 2), hsi_norm)
        # Apply Log_Softmax for KLDiv input Q
        Q_hsi = F.log_softmax(sim_hsi / self.temperature, dim=-1)

        # 3. Compute KL Divergence ---
        # We want HSI structure (Q) to approximate LiDAR structure (P)
        loss = self.kl_div(Q_hsi, P_lidar)

        return loss

