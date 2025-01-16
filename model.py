# model.py

import torch
import torch.nn as nn
import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class Embeddings(nn.Module):
    def __init__(self, input_shape, patch_size=16, embed_dim=768, dropout=0.):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = input_shape[-4]
        self.n_patches = int((input_shape[-1] * input_shape[-2] * input_shape[-3]) / (patch_size * patch_size * patch_size))
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=self.in_channels, out_channels=self.embed_dim,
                                          kernel_size=self.patch_size, stride=self.patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = rearrange(x, "b n h w d -> b (h w d) n")
        # batch, embed_dim, height/patch, width/patch, depth/patch
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0, "d_model is not divisible by n_head"
        self.d_k = d_model // n_head

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.scale = math.sqrt(self.d_k)

        self.dense = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, query, key, value, mask):
        # (Seq_len, d_k) -> (Seq_len, Seq_len)

        matmul_qk = query @ key.transpose(-2, -1)

        scaled_attention_logits = matmul_qk / self.scale

        if mask is not None:
            scaled_attention_logits.masked_fill_(mask == 0, -1e9) # if mask == 0 fill it as -1e9 (sim -inf)

        attention_score = scaled_attention_logits.softmax(dim=-1)
        attention_weights = self.dropout(attention_score)
        x = attention_weights @ value

        return x, attention_score

    def split_heads(self, x, n_head, d_k):
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, n_head, d_k) -> (Batch, n_head, Seq_len, d_k)
        x = x.view(x.shape[0], x.shape[1], n_head, d_k).transpose(1, 2)
        return x

    def forward(self, query, key, value, mask):
        Q = self.query(query) # (Seq_len, d_model) -> (Seq_len, d_model)
        K = self.key(key) # (Seq_len, d_model) -> (Seq_len, d_model)
        V = self.value(value) # (Seq_len, d_model) -> (Seq_len, d_model)

        Q = self.split_heads(Q, self.n_head, self.d_k)
        K = self.split_heads(K, self.n_head, self.d_k)
        V = self.split_heads(V, self.n_head, self.d_k)

        x, attention_score = self.scaled_dot_product_attention(Q, K, V, mask)

        # (Batch, n_head, Seq_len, d_k) -> (Batch, Seq_len, n_head, d_k) -> (Batch, Seq_len, d_model)
        concat_attention  = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_head * self.d_k)

        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        return self.dense(concat_attention), attention_score


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int = 768, dropout: float = 0.1):
        super().__init__()
        self.ff_1 = nn.Linear(d_model, d_model * 4) # w1 and b1
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.ff_2 = nn.Linear(d_model * 4, d_model) # w2 and b2

    def forward(self, x):
        x = self.ff_2(self.dropout(self.relu(self.ff_1(x))))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 8, dropout: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttentionBlock(d_model, n_head, dropout)
        self.feed_forward = FeedForwardBlock(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.layer_norm(x)
        x, attention_score = self.self_attention(x, x, x, None)
        x = res + self.dropout(x)

        res = x
        x = self.layer_norm(x)
        x = self.feed_forward(x)
        x = res + self.dropout(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 8, depth: int = 12, dropout: float = 0.1, extract: list = [3,6,9,12]):
        super().__init__()

        self.ViT = nn.ModuleList([EncoderBlock(d_model, n_head, dropout) for _ in range(depth)])
        self.extract = extract

    def forward(self, x):
        Z_layer = []

        for i, layer in enumerate(self.ViT):
            x = layer(x)
            if i+1 in self.extract:
                Z_layer.append(x)

        return Z_layer


class Upsample3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upsample(x)


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=True):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=((kernel_size - 1) // 2))
        self.act = act
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.act:
            return self.relu(self.batch_norm(self.conv3d(x)))
        else:
            return self.conv3d(x)


class Deconv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.deconv3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv3d = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2))
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv3d(self.deconv3d(x))))


class UNETR(nn.Module):
    def __init__(self, img_shape=(128, 128, 96), patch_size=16, in_channels=2,
                 out_channels=4, d_model=768, n_head=8, dropout=0.1, light_r=4):
        super().__init__()

        self.img_shape = img_shape
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.patch_dim = [int(x / patch_size) for x in img_shape]
        self.conv_channels = [int(i/light_r) for i in [32, 64, 128, 256, 512, 1024]]

        self.embedding = Embeddings((in_channels, *img_shape))

        # U-net encoder
        self.encoder = VisionTransformer()

        # U-net decoder
        self.decoder0 = nn.Sequential(
            Conv3dBlock(in_channels, self.conv_channels[0], 3),
            Conv3dBlock(self.conv_channels[0], self.conv_channels[1], 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv3dBlock(d_model, self.conv_channels[2], 3),
            Deconv3dBlock(self.conv_channels[2], self.conv_channels[2], 3),
            Deconv3dBlock(self.conv_channels[2], self.conv_channels[2], 3)
        )

        self.decoder6 = nn.Sequential(
            Deconv3dBlock(d_model, self.conv_channels[3], 3),
            Deconv3dBlock(self.conv_channels[3], self.conv_channels[3], 3)
        )

        self.decoder9 = nn.Sequential(
            Deconv3dBlock(d_model, self.conv_channels[4], 3)
        )

        self.decoder12_upsampler = Upsample3DBlock(d_model, self.conv_channels[4])

        self.decoder9_upsampler = nn.Sequential(
            Conv3dBlock(self.conv_channels[5], self.conv_channels[3], 3),
            Conv3dBlock(self.conv_channels[3], self.conv_channels[3], 3),
            Conv3dBlock(self.conv_channels[3], self.conv_channels[3], 3),
            Upsample3DBlock(self.conv_channels[3], self.conv_channels[3])
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3dBlock(self.conv_channels[4], self.conv_channels[2], 3),
            Conv3dBlock(self.conv_channels[2], self.conv_channels[2], 3),
            Upsample3DBlock(self.conv_channels[2], self.conv_channels[2])
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3dBlock(self.conv_channels[3], self.conv_channels[1], 3),
            Conv3dBlock(self.conv_channels[1], self.conv_channels[1], 3),
            Upsample3DBlock(self.conv_channels[1], self.conv_channels[1])
        )

        self.decoder0_header = nn.Sequential(
            Conv3dBlock(self.conv_channels[2], self.conv_channels[1], 3),
            Conv3dBlock(self.conv_channels[1], self.conv_channels[1], 3),
            Conv3dBlock(self.conv_channels[1], out_channels, 1, act=False)
        )

    def forward(self, x):
        z0 = x

        # x = (1, 1176, 768)
        x = self.embedding(x)

        # z = (1, 1176, 768)
        z3, z6, z9, z12 = self.encoder(x)

        # z = (1, 768, 14, 14, 6)
        z3 = z3.transpose(-1, -2).view(-1, self.d_model, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.d_model, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.d_model, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.d_model, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return output
