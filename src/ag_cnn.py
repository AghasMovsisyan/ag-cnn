import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(inter_channels, x_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        theta_x = self.theta(x)
        psi_g = self.psi(g)

        if psi_g.shape[2:] != theta_x.shape[2:]:
            psi_g = F.interpolate(
                psi_g,
                size=theta_x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        f = self.relu(theta_x + psi_g)

        att_map = F.softmax(self.phi(f), dim=1)

        out = x + x * att_map
        return out, att_map


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AG_CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dropout_p=0.3):
        super().__init__()

        self.conv1 = conv_block(in_channels, 4)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = conv_block(4, 8)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = conv_block(8, 16)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = conv_block(16, 32)

        self.att1 = AttentionGate(16, 32, inter_channels=4)
        self.att2 = AttentionGate(8, 32, inter_channels=2)
        self.att3 = AttentionGate(4, 32, inter_channels=1)

        self.fc = nn.Sequential(
            nn.Linear(16 + 8 + 4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, return_att=False):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        g = self.conv4(self.pool3(x3))

        o1, att1 = self.att1(x3, g)
        o2, att2 = self.att2(x2, g)
        o3, att3 = self.att3(x1, g)

        f1 = o1.flatten(2).sum(2)
        f2 = o2.flatten(2).sum(2)
        f3 = o3.flatten(2).sum(2)

        features = torch.cat([f1, f2, f3], dim=1)
        logits = self.fc(features)

        if return_att:
            return logits, [att1, att2, att3]

        return logits
