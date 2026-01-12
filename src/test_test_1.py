import torch
import torch.nn as nn
from torch.nn import LSTM, Linear


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


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.column_lstm = ColumnLSTM(
            image_height=80,
            image_channels=3,
            hidden_size=36
        )

        self.attention = Attention(hidden_size=36)
        self.dropout = nn.Dropout(0.2)
        self.fc = Linear(36, 3)

    def forward(self, x):
        x = self.column_lstm(x)         # (B, W, 36)
        x, attn_weights = self.attention(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x, attn_weights


def count_parameters(model, verbose=False):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total trainable parameters: {total}")
    return total


if __name__ == '__main__':
    model = Model()
    image = torch.randn((8, 3, 80, 50))

    output, attn = model(image)

    count_parameters(model, True)
    print("Output:", output)
    print("Output shape:", output.shape)
    print("Attention shape:", attn.shape)  # (B, W, 1)
