import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import Block as TransBlock
from timm.models.vision_transformer import PatchEmbed


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname or 'Embedding' == classname:
        # print("Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
        # if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    if 'LayerNorm' in classname:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)
    # return F.relu(x)

def get_norm_layer(in_channels, mode):
    if mode == 'bn':
        return nn.BatchNorm2d(in_channels)
    elif mode == 'in':
        return nn.InstanceNorm2d(in_channels, affine=False)
    elif mode == 'gn':
        return nn.GroupNorm(num_groups=in_channels//32, num_channels=in_channels)
    elif mode == 'none':
        return nn.Identity()
    else:
        raise ValueError



def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, ndim, nhead, dropout):
        super().__init__()
        assert ndim % nhead == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(ndim, ndim)
        self.query = nn.Linear(ndim, ndim)
        self.value = nn.Linear(ndim, ndim)
        # regularization
        self.drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(ndim, ndim)
        self.n_head = nhead

    def forward(self, q, k, v, mask=None):
        # v: [B, L, C]
        # mask: [B, 1, Lq, Lk]
        B, Lq, C = q.size()
        Lk = k.size(1)
        Lv = v.size(1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(k).view(B, Lk, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lk, hs)
        q = self.query(q).view(B, Lq, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lq, hs)
        v = self.value(v).view(B, Lv, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lv, hs)

        # causal self-attention; Self-attend: (B, nh, Lq, hs) x (B, nh, hs, Lk) -> (B, nh, Lq, Lk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            # mask:[B, 1, Lq, Lk]
            assert mask.dim() == 4
            att = att.masked_fill(mask == 0, float('-inf'))

        if v.dtype == torch.float16:
            att = att.to(torch.float32)
            fp16 = True
        else:
            fp16 = False
        att = F.softmax(att, dim=-1) # (B, nh, Lq, kL*)
        if fp16:
            att = att.to(torch.float16)
        att = self.drop(att)
        y = att @ v  # (B, nh, Lq, Lk*) x (B, nh, Lv, hs) -> (B, nh, Lq, hs)

        y = y.transpose(1, 2).contiguous().view(B, Lq, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

class SelfAttention(nn.Module):
    def __init__(self, ndim, nhead, dropout):
        super().__init__()
        self.self_attn = MultiHeadDotProductAttention(ndim, nhead, dropout=dropout)
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(ndim, ndim * 4)
        self.linear2 = nn.Linear(ndim * 4, ndim)

        self.norm1 = nn.LayerNorm(ndim, eps=1e-12)
        self.norm2 = nn.LayerNorm(ndim, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_pos, mask=None):
        # src: [B, L, C]
        # src_pos: [B, L, C]
        # mask: [B, 1, L, L]
        q = k = self.with_pos_embed(src, src_pos)
        src2 = self.self_attn(q=q, k=k, v=src, mask=mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.gelu(self.linear1(src)))
        src = src + self.dropout2(src2)

        return self.norm2(src)


# condition embedding
class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, num_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, K = x.shape # batch, channel, voxels
        # assert K == self.num_voxels, \
        #     f"Input fmri length ({K}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x # [B, K, C]

class RoIEmbed1D(nn.Module): 
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, roi_patch=[495, 342, 288, 331, 229], patch_size=32, in_chans=1, embed_dim=768):
        super().__init__()
        roi_patch = [0] + roi_patch
        self.roi_patch = torch.cumsum(torch.tensor(roi_patch), dim=0)
        self.layers = nn.ModuleList() 
        for c_in in roi_patch[1:]:
            self.layers.append(nn.Sequential(
                # nn.Conv1d(in_chans, patch_size, kernel_size=1, stride=1),
                nn.Conv1d(in_chans, patch_size, kernel_size=3, stride=1, padding=1),
                nn.Linear(c_in, embed_dim, bias=True),
            ))

    def forward(self, x, **kwargs):
        B, C, K = x.shape # batch, channel, voxels
        assert self.roi_patch[-1] == K
        
        out = []
        for i in range(len(self.roi_patch) - 1):
            z = x[:, :, self.roi_patch[i]:self.roi_patch[i+1]]
            y = self.layers[i](z) # [B, 1, K] -> [B, n, K] -> [B, n, C]
            out.append(y)

        out = torch.cat(out, dim=1)
        return out # [B, K, C]


# image features TO RGB image, decoder only, maskgit pipeline
class MaskGIT_Image2RGB(nn.Module):
    def __init__(self, config):
        super(MaskGIT_Image2RGB, self).__init__()
        self.config = config.Model
        decoder_embed_dim = config.Model['decoder_embed_dim']
        self.num_voxel = config.Data['num_voxel']
        num_latent_size = config.Model['resolution'] // config.Model['patch_size']
        self.num_embed = config.Model['num_codebook']  # + mask_token  
        num_seq_length = num_latent_size ** 2 + 1
        img_dim = 1024

        # decoder
        self.decoder_embed = nn.Linear(img_dim, decoder_embed_dim, bias=True)
        self.token_emb = nn.Embedding(self.num_embed + 1000 + 1, decoder_embed_dim) # [2025, 768]
        self.de_pos_emb = nn.Embedding(num_seq_length, decoder_embed_dim) # [257, 768]
        self.ln = nn.LayerNorm(decoder_embed_dim) # [768]
        self.dropout2 = nn.Dropout(config.Model['drop_rate'])
        self.decoder = nn.ModuleList([SelfAttention(decoder_embed_dim, config.Model['num_heads'], config.Model['drop_rate']) 
                                      for _ in range(config.Model['decoder_depth'])])

        self.mlps = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_embed_dim, eps=1e-12)
        )
        self.mlps_bias = nn.Parameter(torch.zeros(self.num_embed + 1000 + 1)) # [2025]
        self.apply(weights_init)
  
    def forward(self, x, cls):
        # x: [B, L]
        # cls: [B, C]

        # embed tokens
        c = self.decoder_embed(cls[:, None]) # [B, 1, C]
        z = self.token_emb(x) # [B, L, C]
        z = torch.cat([c, z], dim=1) # [B, 1+L, C]
        z_pos = self.de_pos_emb(torch.arange(z.shape[1], dtype=torch.long, device=c.device)[None]) # [1, 1+L, C]
        z = self.dropout2(self.ln(z + z_pos)) # [B, 1+L, C]

        # apply Transformer blocks
        for blk in self.decoder:
            z = blk(z, None) # [B, 1+L, C]

        # pred
        emb = self.mlps(z) # [B, 1+L, C]
        logits = torch.matmul(emb, self.token_emb.weight.T) + self.mlps_bias # [B, 1+L, N+1000+1]

        return logits[:, 1:, :self.num_embed].contiguous() # [B, L, N]


class fMRI_Autoencoder(nn.Module):
    def __init__(self, config):
        super(fMRI_Autoencoder, self).__init__()
        patch_size = config.Data['patch_size']
        roi_patch = config.Data['roi_patch']
        num_voxel = config.Data['num_voxel']
        embed_dim = config.Model['embed_dim']
        decoder_embed_dim = config.Model['decoder_embed_dim']
        num_head = config.Model['num_heads']
        drop_p = config.Model['drop_rate']
        in_chans = 1
        img_dim = 1024

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = RoIEmbed1D(roi_patch, patch_size, in_chans, embed_dim)
        # num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, len(roi_patch) * patch_size + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            TransBlock(embed_dim, num_head, 1.0, qkv_bias=True, 
                       drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config.Model['encoder_depth'])])
        self.norm = nn.LayerNorm(embed_dim)

        self.pred = nn.Linear(embed_dim, img_dim, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(img_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, len(roi_patch) * patch_size + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            TransBlock(decoder_embed_dim, num_head, 1.0, qkv_bias=True, 
                       drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config.Model['decoder_depth'])])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        self.channel_mapper = nn.ModuleList([nn.Sequential(
            nn.Linear(decoder_embed_dim, out_dim, bias=True),
            )
            for out_dim in roi_patch])
        self.decoder_pred = nn.Conv1d(patch_size, 1, kernel_size=3, stride=1, padding=1)

        # --------------------------------------------------------------------------
        self.num_voxel = num_voxel
        self.patch_size = patch_size
        self.roi_patch = roi_patch
        self.initialize_weights()

    def initialize_weights(self, ):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], len(self.roi_patch) * self.patch_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], len(self.roi_patch) * self.patch_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # x: [B, 1, K]
        x = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        z = self.pred(x[:, :1])
        return z
        # return x[:, :1]

    def forward_decoder(self, x):
        # x: [B, 1, C]
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], len(self.roi_patch) * self.patch_size, 1)
        z = torch.cat([x, mask_tokens], dim=1)  # append cls token

        # add pos embed
        z = z + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            z = blk(z) # [B, 1+K, C]
        z = self.decoder_norm(z) # [B, 1+K, C]

        # predictor projection
        z_list = torch.chunk(z[:, 1:], len(self.roi_patch), dim=1)
        out = []
        for i, zz in enumerate(z_list): # [B, k, C]
            y = self.channel_mapper[i](zz) # [B, k, C] -> [B, k, n]
            y = self.decoder_pred(y) # [B, k, n] -> [B, 1, n]
            out.append(y)

        out = torch.cat(out, dim=2) # [B, 1, L]
        return out

    def forward(self, fmri):
        # fmri: [B, 1, K]

        latent = self.forward_encoder(fmri)
        rec = self.forward_decoder(latent)

        return rec

