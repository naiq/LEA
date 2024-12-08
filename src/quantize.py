import torch
import torch.nn as nn
import torch.nn.functional as F



class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.eps = 1e-6

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, mask=None):
        if z.dtype == torch.float16:
            fp16 = True
            z = z.to(torch.float32)
        else:
            fp16 = False
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if fp16:
            z_q = z_q.to(torch.float16)

        # return z_q, (perplexity, min_encodings, min_encoding_indices)   # [B C H W]
        return z_q, (min_encodings, min_encoding_indices)   # [B C H W]

    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    def approximate_codebook(self, probs, shape):
        z_q = torch.matmul(probs.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class MaskGIT_VectorQuantizer(nn.Module):
    def __init__(self, num_code, dim_code, norm=True):
        super(MaskGIT_VectorQuantizer, self).__init__()
        self.eps = 1e-6
        self.norm = norm

        self.embedding = nn.Embedding(num_code, dim_code) # [M, C]
        self.embedding.weight.data.uniform_(-1.0 / num_code, 1.0 / num_code)

    def forward(self, z):
        # z: [N, C, H, W]
        if z.dtype == torch.float16:
            fp16 = True
            z = z.to(torch.float32)
        else:
            fp16 = False

        N, C, H, W = z.shape
        # reshape z and flatten: [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(N*H*W, C)

         # norm
        if self.norm:
            z_flat_norm = F.normalize(z_flat, p=2, dim=1) # [NHW, C]
            code_norm = F.normalize(self.embedding.weight, p=2, dim=1) # [M, C]
            d = torch.sum(z_flat_norm ** 2, dim=1, keepdim=True) + \
                torch.sum(code_norm ** 2, dim=1) - 2 * \
                torch.matmul(z_flat_norm, code_norm.t()) # [NHW, M]
        else:
            d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.matmul(z_flat, self.embedding.weight.t()) # [NHW, M]

        min_idx = torch.argmin(d, dim=1, keepdim=True).repeat(1, C) # [NHW, C]
        quant = self.read_codebook(min_idx, shape=(N, H, W, C)) # [N, C, H, W]

        if fp16:
            quant = quant.to(torch.float16)
        index = min_idx[:, 0].view(N, H, W) # [N, H, W]

        return quant, index # [N, C, H, W], [N, H, W]

    def read_codebook(self, index, shape=None):
        # index: [NHW, C]
        quant = torch.gather(self.embedding.weight, 0, index) # [NHW, C]
        if shape is not None:
            quant = quant.view(shape).permute(0, 3, 1, 2).contiguous() # [N, H, W, C] -> [N, C, H, W]

        return quant

