import torch
from src.ag_cnn import AG_CNN   # քո model ֆայլը

model = AG_CNN()
model.eval()

dummy = torch.randn(1, 3, 128, 128)  # input size-ը քո dataset-ի նման

torch.onnx.export(
    model,
    dummy,  
    "a_gcnn.onnx",
    input_names=["Input"],
    output_names=["Logits"],
    opset_version=18
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # add dropout here
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        

        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()

        self.theta = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.phi   = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)
        self.psi   = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)

        if phi_g.shape[2:] != theta_x.shape[2:]:
            phi_g = F.interpolate(
                phi_g,
                size=theta_x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        f = self.relu(theta_x + phi_g)
        alpha = self.sigmoid(self.psi(f))   # (B,1,H,W)

        out = x * alpha                     # attention gating
        return out, alpha

class RA_GCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dropout_p=0.5):
        super().__init__()

        # Encoder (reduced capacity)
        self.res1 = ResidualConvBlock(in_channels, 8)
        self.pool1 = nn.MaxPool2d(2)

        self.res2 = ResidualConvBlock(8, 16)
        self.pool2 = nn.MaxPool2d(2)

        self.res3 = ResidualConvBlock(16, 32)
        self.pool3 = nn.MaxPool2d(2)

        self.res4 = ResidualConvBlock(32, 48)  # ↓ was 64

        # Attention Gates
        self.att1 = AttentionGate(32, 48, inter_channels=16)
        self.att2 = AttentionGate(16, 48, inter_channels=8)
        self.att3 = AttentionGate(8, 48, inter_channels=4)

        # Regularization after attention
        self.att_drop = nn.Dropout2d(0.25)

        # Classifier (more stable)
        self.fc = nn.Sequential(
            nn.Linear(32 + 16 + 8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_att=False):
        # Encoder
        x1 = self.res1(x)
        x2 = self.res2(self.pool1(x1))
        x3 = self.res3(self.pool2(x2))
        g  = self.res4(self.pool3(x3))

        # Attention
        o1, a1 = self.att1(x3, g)
        o2, a2 = self.att2(x2, g)
        o3, a3 = self.att3(x1, g)

        # ↓ Regularize attention output
        o1 = self.att_drop(o1)
        o2 = self.att_drop(o2)
        o3 = self.att_drop(o3)

        # Global aggregation
        f1 = F.adaptive_avg_pool2d(o1, 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(o2, 1).flatten(1)
        f3 = F.adaptive_avg_pool2d(o3, 1).flatten(1)

        features = torch.cat([f1, f2, f3], dim=1)
        logits = self.fc(features)

        if return_att:
            return logits, [a1, a2, a3]

        return logits
