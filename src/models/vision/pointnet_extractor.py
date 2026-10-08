"""
PointNet Point-Cloud Feature Extraction Module.
Processes 3D point cloud spatial coordinates (N x 3) and extracts a symmetric global 3D feature embedding (64-D).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

class PointNetExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 64):
        super(PointNetExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Shared MLP 1: 3 -> 64 -> 128
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Shared MLP 2: 128 -> 256 -> 512
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Output Feature Projection Head: 512 -> 256 -> out_dim (64)
        self.fc_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D PointNet feature extraction.
        Input:
            pts: Tensor of shape [B, N, 3] or [N, 3]
        Returns:
            z_3d: Normalized 3D point-cloud embedding [B, out_dim]
        """
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)

        B, N, C = pts.shape
        x = pts.transpose(1, 2)

        x = self.mlp1(x)
        x = self.mlp2(x)

        global_feat = torch.max(x, 2)[0]

        out_emb = self.fc_head(global_feat)
        z_3d = F.normalize(out_emb, p=2, dim=-1)
        return z_3d

    def extract_point_cloud_embedding(self, point_array: np.ndarray) -> np.ndarray:
        """
        Convenience function taking numpy array [N, 3] and returning numpy vector [64].
        """
        self.eval()
        with torch.no_grad():
            pts_tensor = torch.tensor(point_array, dtype=torch.float32)
            z_3d = self.forward(pts_tensor)
            return z_3d.cpu().numpy()[0]

def load_point_cloud_fixture(filepath: str, num_points: int = 1024) -> np.ndarray:
    """
    Loads 3D point cloud coordinates from file or generates deterministic sphere/helix coordinates.
    """
    if os.path.exists(filepath) and filepath.endswith(".json"):
        import json
        with open(filepath, "r") as f:
            data = json.load(f)
            pts = np.array(data.get("points", []), dtype=np.float32)
            if len(pts) > 0:
                return pts

    # Deterministic Synthetic 3D Protein Helix Point Cloud Generator
    t = np.linspace(0, 4 * np.pi, num_points)
    x = np.sin(t) + 0.1 * np.random.randn(num_points)
    y = np.cos(t) + 0.1 * np.random.randn(num_points)
    z = t / (4 * np.pi) + 0.1 * np.random.randn(num_points)
    
    pts = np.column_stack([x, y, z]).astype(np.float32)
    return pts
