"""
adapted from https://github.com/ahmedgh970/Transformers_Unsupervised_Anomaly_Segmentation
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale,
                      p2=self.dim_scale, c=C // (self.dim_scale**2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class HTAES(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        patch_size = config.patch_size
        image_size = config.image_size
        self.input_resolution = image_size // patch_size
        self.filters = config.filters

        self.conv1 = nn.Conv2d(config.img_channels, self.filters, patch_size, patch_size)
        self.norm = nn.LayerNorm((self.filters, self.input_resolution, self.input_resolution))
        self.final_expand = FinalPatchExpand_X4(self.input_resolution, self.filters)
        self.conv2 = nn.Conv2d(self.filters, config.img_channels, 1, 1)
        self.tae = HTAE(config)

    def forward(self, x: Tensor):
        b, c, w, h = x.shape
        x = self.conv1(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).view(b, self.input_resolution ** 2, self.filters)
        x = self.tae(x)
        x = self.final_expand(x)
        x = x.permute(0, 2, 1).view(b, self.filters, w, h)
        x = self.conv2(x)
        return x

    def loss(
        self,
        x: Tensor,
        rec: Tensor
    ):
        loss = (x - rec).abs().mean()
        return loss


class HTAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        emb_dim = config.filters
        num_heads = config.num_heads
        dropout = config.dropout
        n_layers = config.transformer_layers
        patch_size = config.patch_size
        image_size = config.image_size
        input_res = image_size // patch_size

        # Transformer Encoder
        self.encoder = TransformerEncoder(input_res, n_layers, emb_dim, num_heads, dropout)
        # Transformer Decoder
        for i in range((n_layers // 2) - 1):
            input_res //= 2
            emb_dim *= 2

        self.decoder = TransformerDecoder(input_res, n_layers, emb_dim, num_heads, dropout)

    def forward(self, x: Tensor):
        x, skip_cons = self.encoder(x)
        x = self.decoder(x, skip_cons)
        return x

    def loss(
        self,
        x: Tensor,
        rec: Tensor
    ):
        loss = (x - rec).abs().mean()
        return loss


class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_res: int,
                 n_layers: int,
                 emb_dim: int,
                 num_heads: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.n_layers = n_layers

        self.blocks = []
        # first block, no patch merging
        self.blocks.append(
            nn.Sequential(
                TransformerEncoderBlock(emb_dim, num_heads, dropout),
                TransformerEncoderBlock(emb_dim, num_heads, dropout)
            ))

        for i in range((self.n_layers // 2) - 1):
            patch_merge = PatchMerging(input_res, emb_dim)
            input_res //= 2
            emb_dim *= 2

            transformer_block = nn.Sequential(
                TransformerEncoderBlock(emb_dim, num_heads, dropout),
                TransformerEncoderBlock(emb_dim, num_heads, dropout)
            )
            self.blocks.append(nn.Sequential(*[patch_merge, transformer_block]))

        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x: Tensor):

        skip_cons = []

        for block in self.blocks:
            x = block(x)

            skip_cons.append(x)

        return x, skip_cons


class TransformerDecoder(nn.Module):
    def __init__(self,
                 input_res: int,
                 n_layers: int,
                 emb_dim: int,
                 num_heads: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.n_layers = n_layers

        self.trans_blocks = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.expand_layers = nn.ModuleList()

        # first block, no patch merging
        self.first_block = TransformerDecoderBlock(emb_dim, num_heads, dropout)

        for i in range((self.n_layers // 2) - 1):
            self.expand_layers.append(PatchExpand(input_res, emb_dim))
            input_res *= 2
            emb_dim //= 2
            self.norm_layers.append(nn.LayerNorm(emb_dim))
            self.trans_blocks.append(TransformerDecoderBlock(emb_dim, num_heads, dropout))

    def forward(self, x: Tensor, skip_cons: list):
        x = self.first_block(x, x)
        for i in range((self.n_layers // 2) - 1):
            x = self.expand_layers[i](x)
            x = skip_cons[-i - 2] + x
            x = self.norm_layers[i](x)
            x = self.trans_blocks[i](x, x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int = 12,
                 dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.att = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True)
        self.att_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(emb_dim)

    def forward(self, x: Tensor):

        res1 = x
        x = self.norm1(x)
        x = self.att(x, x, x, need_weights=False)[0]
        x = self.att_drop(x)
        x = x + res1
        res2 = x
        x = self.mlp(self.norm2(x))
        x = x + res2
        x = self.norm3(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.att1 = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True)
        self.att_drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.att2 = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True)
        self.att_drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(emb_dim)

    def forward(self, x: Tensor, orig_patches: Tensor):

        res1 = x
        x = self.att1(x, x, x, need_weights=False)[0]
        x = self.att_drop1(x)
        x = x + res1
        x = self.norm1(x)
        res2 = x
        x = self.att2(x, orig_patches, orig_patches, need_weights=False)[0]
        x = self.att_drop2(x)
        x = x + res2
        x = self.norm2(x)
        res3 = x
        x = self.mlp(x)
        x = x + res3
        x = self.norm3(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.layer1 = TransformerDecoderLayer(emb_dim, num_heads, dropout)
        self.layer2 = TransformerDecoderLayer(emb_dim, num_heads, dropout)

    def forward(self, x: Tensor, orig_patches: Tensor):

        x = self.layer1(x, orig_patches)
        x = self.layer2(x, orig_patches)
        return x


if __name__ == '__main__':
    from argparse import Namespace
    config = Namespace(
        patch_size=4,
        image_size=128,
        num_heads=8,
        transformer_dropout=0.1,
        transformer_layers=12,
        filters=96,
        img_channels=1,
        dropout=0
    )
    x = torch.randn(2, 1, 128, 128)
    model = HTAES(config)
    y = model(x)
    print(y.shape)
