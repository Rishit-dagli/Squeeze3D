import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerCNN(nn.Module):
    def __init__(self, input_shape, output_size):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_shape[0] // 4) * (input_shape[1] // 4), 512)
        self.fc2 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ThreeLayerConvMLP(nn.Module):
    def __init__(self, input_shape, output_size, dropout_prob=0.0):
        super(ThreeLayerConvMLP, self).__init__()
        self.input_shape = input_shape

        self.hidden_channels = 64

        h, w = input_shape
        h_out = (h + 2 - 3) // 2 + 1
        w_out = (w + 2 - 3) // 2 + 1
        h_out = (h_out + 2 - 3) // 2 + 1
        w_out = (w_out + 2 - 3) // 2 + 1

        self.conv1 = nn.Conv2d(
            1, self.hidden_channels, kernel_size=3, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.conv2 = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(self.hidden_channels * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.flat_size = (self.hidden_channels * 2) * h_out * w_out

        self.fc1 = nn.Linear(self.flat_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, 1, self.input_shape[0], self.input_shape[1])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = x.view(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.gelu1 = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu2(x)

        if x.shape == identity.shape:
            x = x + identity
        return x


class ChunkedLinear(nn.Module):
    def __init__(self, in_features, out_features, chunk_size=1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_size = chunk_size

        self.num_chunks = (out_features + chunk_size - 1) // chunk_size
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features, min(chunk_size, out_features - i * chunk_size))
                for i in range(self.num_chunks)
            ]
        )

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            outputs.append(layer(x))
        return torch.cat(outputs, dim=-1)


class HighDimensionalUNet(nn.Module):
    def __init__(self, input_shape, output_size, base_channels=64, dropout_prob=0.0):
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size

        if len(input_shape) == 2:
            self.input_size = input_shape[0] * input_shape[1]
        elif len(input_shape) == 3:
            self.input_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.reshape_channels = base_channels
        self.reshape_size = int((self.input_size / base_channels) ** 0.5)

        self.encoder1 = UNetBlock(base_channels, base_channels, dropout_prob)
        self.encoder2 = UNetBlock(base_channels, base_channels * 2, dropout_prob)
        self.encoder3 = UNetBlock(base_channels * 2, base_channels * 4, dropout_prob)

        self.bottleneck = UNetBlock(base_channels * 4, base_channels * 4, dropout_prob)

        self.decoder3 = UNetBlock(base_channels * 8, base_channels * 2, dropout_prob)
        self.decoder2 = UNetBlock(base_channels * 4, base_channels, dropout_prob)
        self.decoder1 = UNetBlock(base_channels * 2, base_channels, dropout_prob)

        flatten_size = base_channels * self.reshape_size * self.reshape_size
        self.final_reshape = nn.Sequential(
            nn.Flatten(), ChunkedLinear(flatten_size, output_size, chunk_size=1024)
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = x.view(
            batch_size, self.reshape_channels, self.reshape_size, self.reshape_size
        )

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))

        b = self.bottleneck(self.maxpool(e3))

        d3 = self.decoder3(torch.cat([self.upsample(b), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))

        output = self.final_reshape(d1)
        return output
