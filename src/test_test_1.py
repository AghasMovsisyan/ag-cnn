import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, Linear

# ------------------- Column LSTM -------------------
class ColumnLSTM(nn.Module):
    def __init__(self, image_height, image_channels=3, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=image_channels * image_height,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2)      # (B, W, C, H)
        x = x.reshape(B, W, C * H)     # (B, W, C*H)
        lstm_out, _ = self.lstm(x)     # (B, W, hidden)
        return lstm_out

# ------------------- Simple LSTM Attention -------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, H)
        weights = self.attn(x)                 # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        context = (x * weights).sum(dim=1)     # (B, H)
        return context, weights

# ------------------- Attention Gate -------------------
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
            psi_g = F.interpolate(psi_g, size=theta_x.shape[2:], mode="bilinear", align_corners=False)

        f = self.relu(theta_x + psi_g)
        att_map = torch.softmax(self.phi(f), dim=1)
        out = x + x * att_map
        return out, att_map

# ------------------- Combined Model -------------------
class ColumnLSTM_AG(nn.Module):
    def __init__(self, image_height=80, image_channels=3, lstm_hidden=36, num_classes=3):
        super().__init__()
        self.column_lstm = ColumnLSTM(image_height=image_height, image_channels=image_channels, hidden_size=lstm_hidden)
        self.lstm_attention = Attention(lstm_hidden)
        self.dropout = nn.Dropout(0.2)

        # Small CNN layers to generate feature maps for attention gates
        self.conv1 = nn.Conv2d(image_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # 3 Attention Gates
        self.att1 = AttentionGate(16, 16, inter_channels=4)
        self.att2 = AttentionGate(8, 16, inter_channels=2)
        self.att3 = AttentionGate(4, 16, inter_channels=1)

        # Fully connected layer after combining all features
        self.fc = Linear(lstm_hidden + 16 + 8 + 4, num_classes)

    def forward(self, x):
        # --- Column LSTM branch ---
        lstm_out = self.column_lstm(x)       # (B, W, lstm_hidden)
        lstm_feat, _ = self.lstm_attention(lstm_out)
        lstm_feat = self.dropout(lstm_feat)

        # --- CNN branch for Attention Gates ---
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))
        g = x3  # Global feature for gating

        o1, att1 = self.att1(x3, g)
        o2, att2 = self.att2(x2, g)
        o3, att3 = self.att3(x1, g)

        f1 = o1.flatten(2).sum(2)
        f2 = o2.flatten(2).sum(2)
        f3 = o3.flatten(2).sum(2)

        features = torch.cat([lstm_feat, f1, f2, f3], dim=1)
        logits = self.fc(features)

        return logits, [att1, att2, att3]

# ------------------- Testing -------------------
if __name__ == "__main__":
    model = ColumnLSTM_AG()
    image = torch.randn((8, 3, 80, 50))
    output, att_maps = model(image)

    print("Output shape:", output.shape)
    print("Attention maps shapes:", [a.shape for a in att_maps])
