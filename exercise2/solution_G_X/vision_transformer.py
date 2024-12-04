import math
from functools import partial

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from util.vision_transformer_util import bcolors

###################### TO-DO #########################
########### Vision Transformer (20 Points) ###########
# 1. Implement patch embedding, positional embedding, and class token (5 Points)
#   --> See: self.patch_embed = None
#   --> See: self.cls_token = None
#   --> See: self.pos_embed = None
# 2. Implement the prepare_tokens function (5 Points)
#   --> See: prepare_tokens()
# 3. Implement the attention class (10 Points)
#   --> See: Attention()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
    ):
        super().__init__()
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        pass
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def forward(self, x):
        """
        Forward pass of the MultiHeadAttention layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, C), where
                B is the batch size, N is the number of elements (e.g., sequence length),
                and C is the feature dimension.

        Returns:
            torch.Tensor: Output tensor after attention mechanism.
            torch.Tensor: Attention weights. (For visualization purposes)
        """
        B, N, C = x.shape
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        pass
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer
    """

    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        pool="cls",
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = dim

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        self.patch_embed = None
        self.cls_token = None
        self.pos_embed = None
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.pool = pool

        # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(dim)

        # Classifier head
        self.mlp_head = nn.Linear(dim, num_classes)

    def prepare_tokens(self, x):
        """
        Prepares input tokens for a transformer model.

        Args:
            x (torch.Tensor): Input tensor with shape (B, nc, w, h).

        Returns:
            torch.Tensor: Processed tokens with positional encoding and [CLS] token.

        Note:
            Assumes the presence of attributes: patch_embed, cls_token, pos_embed.
        """
        B, nc, w, h = x.shape

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # apply patch linear embedding
        pass
        # add the [CLS] token to the embed patch tokens
        pass
        # add positional encoding to each token
        pass
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.norm(x)
        x = self.mlp_head(x)
        out = torch.sigmoid(x)
        return out

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def predict(self, dataloader, device):
        """Predict all samples from a dataloader

        Args:
            dataloader (torch.utils.data.dataloader.DataLoader): PyTorch Dataloader
            device (torch.device or str): "cpu" or "cuda"

        Returns:
            dataframe (pandas dataframe): Table of all predictions
            accuracy (float): Percentage of correct predictions
        """

        array_score = []
        acc = 0

        self.eval()

        for _, (data, label) in enumerate(dataloader, 0):
            data = data.to(device)
            label = label.to(device)
            predictions_raw = self.forward(data)

            predictions_raw_numpy = predictions_raw[:, 0].cpu().detach().numpy()
            predictions = predictions_raw_numpy.copy()
            predictions = predictions > 0.5

            for i_pred, pred in enumerate(predictions):
                if int(predictions[i_pred]) == int(label[i_pred].cpu().numpy()):
                    correct = bcolors.OKGREEN + "✔" + bcolors.ENDC
                else:
                    correct = bcolors.FAIL + "✖" + bcolors.ENDC

                if pred == False:
                    array_score.append(
                        [
                            "Circle",
                            int(predictions[i_pred]),
                            round(predictions_raw_numpy[i_pred], 3),
                            int(label[i_pred].cpu().numpy()),
                            correct,
                        ]
                    )
                elif pred == True:
                    array_score.append(
                        [
                            "Square",
                            int(predictions[i_pred]),
                            round(predictions_raw_numpy[i_pred], 3),
                            int(label[i_pred].cpu().numpy()),
                            correct,
                        ]
                    )

            acc += np.average((label.cpu().numpy() - predictions) == 0)

        dataframe = pd.DataFrame(
            array_score, columns=["class", "Pred", "RAW_Pred", "GT", "Correct?"]
        )

        return dataframe, acc / len(dataloader)
