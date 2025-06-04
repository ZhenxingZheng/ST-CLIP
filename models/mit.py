import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import sys

sys.path.append("../")
from clip.model import QuickGELU


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T=8, embed_dim=512, layers=1, ):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        ori_x = x
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)
        x = x.type(ori_x.dtype) + ori_x

        return x.mean(dim=1, keepdim=False)

class LSTM_cls(nn.Module):
    def __init__(self, T=8, embed_dim=512):
        super().__init__()
        self.T = T
        self.lstm_visual = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                               batch_first=True, bidirectional=False, num_layers=1)

    def forward(self, x):
        x_original = x
        self.lstm_visual.flatten_parameters()
        x, _ = self.lstm_visual(x.float())
        self.lstm_visual.flatten_parameters()
        x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
        x = x.type(x_original.dtype) + x_original

        return x.mean(dim=1, keepdim=False)

class temp_conv1d(nn.Module):
    def __init__(self, T=8, embed_dim=512):
        super().__init__()
        self.shift = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
        weight = torch.zeros(embed_dim, 1, 3)
        weight[:embed_dim // 4, 0, 0] = 1.0
        weight[embed_dim // 4:embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
        weight[-embed_dim // 4:, 0, 2] = 1.0
        self.shift.weight = nn.Parameter(weight)

    def forward(self, x):
        x_original = x
        x = x.permute(0, 2, 1)
        x = self.shift(x.float())
        x = x.permute(0, 2, 1)
        x = x.type(x_original.dtype) + x_original

        return x

if __name__ == '__main__':
    fake_data = torch.randn(2, 8, 512)
    embed_dim = 512
    # net = MultiframeIntegrationTransformer()
    # net = LSTM_cls()
    net = temp_conv1d()

    output = net(fake_data)
    print(output.size())
