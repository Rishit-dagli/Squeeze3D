import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
import math
from typing import Callable, Tuple, List, Union, Optional


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]

        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb(q, k, cos, sin):
    q = rearrange(q, "b h n d -> b n h d")
    k = rearrange(k, "b h n d -> b n h d")

    cos = cos[:, :, : q.shape[2], :]
    sin = sin[:, :, : q.shape[2], :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = rearrange(q_embed, "b n h d -> b h n d")
    k_embed = rearrange(k_embed, "b n h d -> b h n d")

    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, flash_attn=False):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.flash_attn = flash_attn

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(
            qkv, "b n (three h d) -> three b h n d", three=3, h=self.num_heads
        )
        q, k, v = qkv.unbind(0)

        cos, sin = self.rotary(x, seq_len=N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.flash_attn and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            x = torch.matmul(attn, v)

        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1, flash_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, flash_attn)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LT2Transformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_size: int,
        dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        flash_attn: bool = False,
    ):
        super().__init__()
        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]
        else:
            raise ValueError("Unsupported input shape")

        self.input_shape = input_shape
        self.flatten = nn.Flatten()

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_size), nn.Linear(input_size, dim)
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    flash_attn=flash_attn,
                )
                for _ in range(depth)
            ]
        )

        self.output_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_size))

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = self.flatten(x)
        x = self.input_proj(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        for block in self.blocks:
            x = block(x, mask)

        x = x[:, 0]
        x = self.output_proj(x)

        return x


class TwoLayerMLP(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size):
        super(TwoLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        input_size = input_shape[0] * input_shape[1]
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class ResidualNN(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size, num_residual_blocks):
        super(ResidualNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_in = nn.Linear(input_shape[0] * input_shape[1], hidden_size)
        self.relu = nn.ReLU()
        self.residual_blocks = nn.ModuleList(
            [self._residual_block(hidden_size) for _ in range(num_residual_blocks)]
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def _residual_block(self, hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc_in(x))
        for block in self.residual_blocks:
            x = self.relu(x + block(x))
        x = self.fc_out(x)
        return x


class ThreeLayerMLP(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size, dropout_prob=0.0):
        super(ThreeLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        input_size = input_shape[0] * input_shape[1]
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class ConstMLP(nn.Module):
    def __init__(self, input_shape, output_size, dim=768, depth=6, dropout=0.1):
        super(ConstMLP, self).__init__()

        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]
        else:
            raise ValueError("Unsupported input shape")

        self.flatten = nn.Flatten()

        self.ln = nn.LayerNorm(input_size)

        self.fc1 = nn.Linear(input_size, dim)
        self.gelu = nn.GELU()

        self.dropout = nn.Dropout(dropout)
        self.network = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth - 2)])

        self.fc2 = nn.Linear(dim, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.ln(x)

        x = self.gelu(self.fc1(x))
        residual = x

        for i, layer in enumerate(self.network):
            new_x = layer(x)
            new_x = self.gelu(new_x)
            if i in [2, 5, 8, 11]:
                new_x = new_x + residual
                residual = new_x
                new_x = self.dropout(new_x)
            x = new_x

        x = self.fc2(x)
        return x


class LT2(nn.Module):
    def __init__(self, input_shape, output_size, hidden_size=2048, dropout_prob=0.0):
        super(LT, self).__init__()

        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]
        else:
            raise ValueError("Unsupported input shape")

        self.flatten = nn.Flatten()

        self.ln = nn.LayerNorm(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout_prob)
        )

        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())

        self.final_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.ln(x)

        identity = x
        x = self.fc1(x)
        x = x + identity

        identity = x
        x = self.fc2(x)
        x = x + identity

        x = self.final_layer(x)
        return x


class LT(nn.Module):
    def __init__(self, input_shape, output_size, hidden_size=2048, dropout_prob=0.0):
        super(LT, self).__init__()
        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        if len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)

        identity = self.fc1(x)
        x = self.ln1(identity)
        x = F.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x + identity)
        x = F.gelu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


class LTOrtho(nn.Module):
    def __init__(self, input_shape, output_size, hidden_size=2048, dropout_prob=0.0):
        super(LTOrtho, self).__init__()
        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        if len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)

        identity = self.fc1(x)
        x = self.ln1(identity)
        x = F.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        mid = x
        x = self.ln2(x + identity)
        x = F.gelu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x, mid


class SimpleProjection(nn.Module):
    def __init__(self, input_shape, output_size):
        super().__init__()

        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]
        else:
            raise ValueError("Unsupported input shape")

        self.flatten = nn.Flatten()
        self.projection = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        return self.projection(x)


class LTpc(nn.Module):
    def __init__(
        self,
        input_shape: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]] = (1024,),
        output_size: int = 8320,
        hidden_size: int = 2048,
        num_layers: int = 12,
        dropout_prob: float = 0.0,
        activation: Callable = F.gelu,
    ):
        super().__init__()

        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]
        else:
            input_size = input_shape[0]

        self.flatten = nn.Flatten()
        self.activation = activation
        self.num_layers = num_layers

        self.first_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.output_layer = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = self.first_layer(x)

        x = self.layer_norm(x)
        residual = x

        x = self.activation(x)
        x = self.dropout(x)

        for i, layer in enumerate(self.hidden_layers):
            layer_input = x
            x = layer(x)

            if i == 0:
                x = x + residual
            elif i % 4 == 0:
                x = x + layer_input

            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)

        return x


class LTpcOrtho(nn.Module):
    def __init__(
        self,
        input_shape: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]] = (1024,),
        output_size: int = 8320,
        hidden_size: int = 2048,
        num_layers: int = 12,
        dropout_prob: float = 0.0,
        activation: Callable = F.gelu,
    ):
        super().__init__()

        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]
        else:
            input_size = input_shape[0]

        self.flatten = nn.Flatten()
        self.activation = activation
        self.num_layers = num_layers

        self.first_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.output_layer = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.flatten(x)

        x = self.first_layer(x)

        x = self.layer_norm(x)
        residual = x

        x = self.activation(x)
        x = self.dropout(x)

        # Store intermediate representation for orthogonal regularization
        mid = None

        for i, layer in enumerate(self.hidden_layers):
            layer_input = x
            x = layer(x)

            if i == 0:
                x = x + residual
            elif i % 4 == 0:
                x = x + layer_input

            # Capture intermediate representation at middle layer
            if i == len(self.hidden_layers) // 2:
                mid = x

            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)

        return x, mid


class LTNeRF(nn.Module):
    def __init__(self, in_channels=96, expansion_factor=4, dropout=0.1):
        super().__init__()

        mid_channels = in_channels * 2

        bottleneck_channels = 24
        expanded_channels = in_channels * expansion_factor

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            LayerNorm3d(mid_channels),
            nn.GELU(),
            ResidualBlock3D(mid_channels, mid_channels * 2),
            nn.Dropout3d(dropout),
            ResidualBlock3D(mid_channels * 2, expanded_channels, stride=2),
            nn.Dropout3d(dropout),
            ResidualBlock3D(expanded_channels, expanded_channels),
            ResidualBlock3D(expanded_channels, expanded_channels // 2, stride=2),
            nn.Dropout3d(dropout),
            nn.Conv3d(expanded_channels // 2, bottleneck_channels, 3, padding=1),
            LayerNorm3d(bottleneck_channels),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(bottleneck_channels, expanded_channels // 2, 3, padding=1),
            LayerNorm3d(expanded_channels // 2),
            nn.GELU(),
            ResidualBlock3D(expanded_channels // 2, expanded_channels),
            nn.Dropout3d(dropout),
            nn.ConvTranspose3d(
                expanded_channels, expanded_channels, 4, stride=2, padding=1
            ),
            LayerNorm3d(expanded_channels),
            nn.GELU(),
            ResidualBlock3D(expanded_channels, expanded_channels),
            nn.Dropout3d(dropout),
            nn.ConvTranspose3d(
                expanded_channels, mid_channels * 2, 4, stride=2, padding=1
            ),
            LayerNorm3d(mid_channels * 2),
            nn.GELU(),
            ResidualBlock3D(mid_channels * 2, mid_channels),
            ResidualBlock3D(mid_channels, in_channels),
        )

        self.enable_skip_connections = True

    def forward(self, x):

        features = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if isinstance(layer, ResidualBlock3D):
                features.append(x)

        encoded = x

        x = encoded
        feature_idx = len(features) - 2

        for i, layer in enumerate(self.decoder):
            x = layer(x)

            if (
                self.enable_skip_connections
                and isinstance(layer, ResidualBlock3D)
                and feature_idx >= 0
            ):
                if x.shape[2:] == features[feature_idx].shape[2:]:
                    x = x + features[feature_idx]
                    feature_idx -= 1

        return x


class LTNeRFOrtho(nn.Module):
    def __init__(self, in_channels=96, expansion_factor=4, dropout=0.1):
        super().__init__()

        mid_channels = in_channels * 2

        bottleneck_channels = 24
        expanded_channels = in_channels * expansion_factor

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            LayerNorm3d(mid_channels),
            nn.GELU(),
            ResidualBlock3D(mid_channels, mid_channels * 2),
            nn.Dropout3d(dropout),
            ResidualBlock3D(mid_channels * 2, expanded_channels, stride=2),
            nn.Dropout3d(dropout),
            ResidualBlock3D(expanded_channels, expanded_channels),
            ResidualBlock3D(expanded_channels, expanded_channels // 2, stride=2),
            nn.Dropout3d(dropout),
            nn.Conv3d(expanded_channels // 2, bottleneck_channels, 3, padding=1),
            LayerNorm3d(bottleneck_channels),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(bottleneck_channels, expanded_channels // 2, 3, padding=1),
            LayerNorm3d(expanded_channels // 2),
            nn.GELU(),
            ResidualBlock3D(expanded_channels // 2, expanded_channels),
            nn.Dropout3d(dropout),
            nn.ConvTranspose3d(
                expanded_channels, expanded_channels, 4, stride=2, padding=1
            ),
            LayerNorm3d(expanded_channels),
            nn.GELU(),
            ResidualBlock3D(expanded_channels, expanded_channels),
            nn.Dropout3d(dropout),
            nn.ConvTranspose3d(
                expanded_channels, mid_channels * 2, 4, stride=2, padding=1
            ),
            LayerNorm3d(mid_channels * 2),
            nn.GELU(),
            ResidualBlock3D(mid_channels * 2, mid_channels),
            ResidualBlock3D(mid_channels, in_channels),
        )

        self.enable_skip_connections = True

    def forward(self, x):

        features = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if isinstance(layer, ResidualBlock3D):
                features.append(x)

        encoded = x

        # Store bottleneck representation for orthogonal regularization
        bottleneck_features = encoded.reshape(encoded.size(0), -1)

        x = encoded
        feature_idx = len(features) - 2

        for i, layer in enumerate(self.decoder):
            x = layer(x)

            if (
                self.enable_skip_connections
                and isinstance(layer, ResidualBlock3D)
                and feature_idx >= 0
            ):
                if x.shape[2:] == features[feature_idx].shape[2:]:
                    x = x + features[feature_idx]
                    feature_idx -= 1

        return x, bottleneck_features


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):

        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)

        x = x.permute(0, 4, 1, 2, 3)
        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, padding=1)
        self.norm1 = LayerNorm3d(out_channels)

        mid_channels = out_channels * 2
        self.conv2 = nn.Conv3d(out_channels, mid_channels, 1)
        self.norm2 = LayerNorm3d(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, 3, padding=1)
        self.norm3 = LayerNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride),
                LayerNorm3d(out_channels),
            )

        self.act = nn.GELU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        residual = out
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.norm3(out)

        out += self.shortcut(x)
        out = self.act(out)

        out += residual
        out = self.act(out)

        return out
