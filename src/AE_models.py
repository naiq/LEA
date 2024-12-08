import torch
import torch.nn as nn
import os
from src.unet_layers import *
from src.quantize import *
from utils.utils import torch_init_model, Config
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm



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


def calculate_adaptive_weight(nll_loss, g_loss, last_layer, weight):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * weight
    return d_weight


# MaskGIT VQModel
class MaskGIT_VQModel(nn.Module):
    def __init__(self, config):
        super(MaskGIT_VQModel, self).__init__()
        ch_latent = config['ch_latent']
        use_spectral_norm = config['use_spectral_norm']
        use_preconv = config['use_preconv']
        num_codebook = config['num_codebook']
        dim_codebook = config['dim_codebook']
        norm_mode = config['norm_mode']
        assert norm_mode == 'gn'
        assert ~use_spectral_norm
        assert ~use_preconv

        self.encoder = MaskGIT_Encoder(config, sample_with_conv=False)
        self.decoder = MaskGIT_Decoder(config, sample_with_conv=True)
        self.quantizer = MaskGIT_VectorQuantizer(num_codebook, dim_codebook, norm=config['norm_codebook'])

        if use_preconv:
            if use_spectral_norm:
                self.pre_conv = SpectralNorm(nn.Conv2d(ch_latent, dim_codebook, kernel_size=1))
                self.post_conv = SpectralNorm(nn.Conv2d(dim_codebook, ch_latent, kernel_size=1))
            else:
                self.pre_conv = nn.Conv2d(ch_latent, dim_codebook, kernel_size=1)
                self.post_conv = nn.Conv2d(dim_codebook, ch_latent, kernel_size=1)
        else:
            assert ch_latent == dim_codebook
            self.pre_conv = nn.Identity()
            self.post_conv = nn.Identity()

    # for transformer
    def encode(self, x):
        z = self.encoder.forward_oneway(x)

        # preconv
        code = self.pre_conv(z)

        # quantize
        quant, indices = self.quantizer(code)

        return quant, indices

    # for transformer
    def decode(self, quant):
        post = self.post_conv(quant)
        rec = self.decoder.forward_oneway(post)

        return rec

    def forward(self, src, tgt):
        # encode
        src_z, tgt_z = self.encoder(src, tgt)

        # preconv
        src_code = self.pre_conv(src_z)
        tgt_code = self.pre_conv(tgt_z)

        # quantize & post_code
        src_quant, _ = self.quantizer(src_code)
        tgt_quant, _ = self.quantizer(tgt_code)

        # postconv
        src_quant_ = src_code + (src_quant - src_code).detach()
        tgt_quant_ = tgt_code + (tgt_quant - tgt_code).detach()
        src_post = self.post_conv(src_quant_)
        tgt_post = self.post_conv(tgt_quant_)

        # decoder
        src_rec, tgt_rec = self.decoder(src_post, tgt_post)

        return src_code, src_quant, src_rec, tgt_code, tgt_quant, tgt_rec

    def forward_oneway(self, input):
        # encode
        z = self.encoder.forward_oneway(input)

        # preconv
        code = self.pre_conv(z)

        # quantize & post_code
        quant, _ = self.quantizer(code)

        # postconv
        quant_ = code + (quant - code).detach()
        post = self.post_conv(quant_)

        # decoder
        rec = self.decoder.forward_oneway(post)

        return code, quant, rec

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

