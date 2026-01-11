import torch
import torch.nn as nn
import numpy as np
import random
from einops import rearrange, repeat
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, lidar):
        x2, lidar = self.fn(x, lidar)
        return x2 + x, lidar
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, lidar):

        return self.fn(self.norm(x), self.norm(lidar))
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, lidar):
        return self.net(x), lidar
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.linear_merge = nn.Linear(dim_head, dim_head)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, x, lidar):
        lidar_out = lidar

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]
       
        q = rearrange(q, 'b n (h d) -> b h n d',h=h)
        k = rearrange(k, 'b n (h d) -> b h n d',h=h)
        v = rearrange(v, 'b n (h d) -> b h n d',h=h)
        lidar = rearrange(lidar, 'b n (h d) -> b h n d',h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        lidar_matrix = torch.matmul(lidar, lidar.transpose(2, 3)) * self.scale
        lidar_matrix = lidar_matrix.softmax(dim=-1)

        dots_reshaped = dots.view(b*h, 1, n, n)
        lidar_reshaped = lidar_matrix.view(b*h, 1, n, n)

        combined = torch.cat([dots_reshaped, lidar_reshaped], dim=1)  # [b*h, 2, n, n]
        mid = self.conv(combined)  # [b*h, 1, n, n]
        dots = mid.view(b, h, n, n) 

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = self.linear_merge(out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, lidar_out

class Transformer(nn.Module):
    def __init__(self, image_size, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()
        self.num_channel = num_channel
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 1):
            self.skipcat.append(nn.Conv2d(self.num_channel + 1, self.num_channel + 1, [1, 2], 1, 0))

        self.lidar_linear = nn.Linear(image_size ** 2, dim)
        self.cie = MCIE(self.num_channel + 1, self.num_channel + 1, 9)

    def forward(self, x, lidar=None):
        last_output = []
        lidar = torch.cat([lidar] * int(self.num_channel+1), dim=1)
        nl = 0
        lidar = self.cie(lidar)
        lidar = lidar.reshape(lidar.shape[0], lidar.shape[1], lidar.shape[2] * lidar.shape[3])
        lidar = self.lidar_linear(lidar)
        for attn, ff in self.layers:
            last_output.append(x)
            if nl > 0:
                x = self.skipcat[nl - 1](torch.cat([x.unsqueeze(3), last_output[nl - 1].unsqueeze(3)], dim=3)).squeeze(3)
            x, lidar = attn(x, lidar)
            x, lidar = ff(x, lidar)
            nl += 1

        return x

class LSLF(nn.Module):
    def __init__(self, image_size, near_band, num_band, total_band, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=16, dropout=0.1, emb_dropout=0.5):
        super().__init__()
        self.near_band = near_band

        self.lidar_embedding = nn.Linear(image_size ** 2, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1,  num_band+ 2, dim))
        self.patch_to_embedding = nn.Linear(image_size ** 2*near_band, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.h2m = H2M(total_band,num_band)
        self.cie = MCIE(3, 1, 3)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(image_size, dim, depth, heads, dim_head, mlp_dim, dropout, num_band+1)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def gain_neighborhood_band(self, x, band_patch):
        N, B, P1, P2 = x.shape
        nn = band_patch // 2

        # 构造带有镜像的 band 索引
        offsets = torch.arange(-nn, nn + 1, device=x.device)  # [-nn, ..., 0, ..., nn]
        band_indices = (torch.arange(B, device=x.device)[None, :] + offsets[:, None]) % B  # (band_patch, B)

        x_group_band = x[:, band_indices, :, :]
        x_group_band = x_group_band.permute(0, 2, 3, 4, 1)
        x_group_band = x_group_band.reshape(N, B, P1 * P2 * band_patch)

        return x_group_band

    def forward(self, x, l):
        x = self.h2m(x)
        x_loss = x
        lidar = l.repeat(1, 3, 1, 1)
        lidar = self.cie(lidar)
        l =lidar
        x = self.gain_neighborhood_band(x, self.near_band)
        x = self.patch_to_embedding(x)  # [batch,n,dim]
        lidar = lidar.reshape(lidar.shape[0], lidar.shape[1], lidar.shape[2] * lidar.shape[3])
        lidar = self.lidar_embedding(lidar)
        x = torch.cat((x, lidar), 1)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, lidar=l)
        x = self.to_latent(x[:, 0])
        x = self.mlp_head(x)
        l = torch.cat([l] * (n-1), dim=1)

        return x, x_loss, l

class H2M(nn.Module):
    def __init__(self, hs_bands, ms_bands):
        super().__init__()

        srf = torch.ones([ms_bands, hs_bands, 1, 1]) * (1.0 / hs_bands)
        self.srf = nn.Parameter(srf)

    def forward(self, hr_hsi):
        srf_div = torch.sum(self.srf, dim=1, keepdim=True)
        srf_div = torch.div(1.0, srf_div)
        srf_div = torch.transpose(srf_div, 0, 1)

        hr_msi = F.conv2d(hr_hsi, self.srf, None)
        hr_msi = torch.mul(hr_msi, srf_div)
        hr_msi = torch.clamp(hr_msi, 0.0, 1.0)

        return hr_msi
    
class MCIE(nn.Module):
    def __init__(self, input_channel, output_channel, mid_channel):
        super().__init__()

        self.begin = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        branch_base = mid_channel // 3
        remainder = mid_channel % 3
        ch1 = branch_base + (1 if remainder > 0 else 0)
        ch2 = branch_base + (1 if remainder > 1 else 0)
        ch3 = branch_base

        self.branch1 = nn.Sequential(
            nn.Conv2d(ch1, ch1, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(ch2, ch2, kernel_size=3, padding=1, bias=True),  # pad=1 keeps output size
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(ch3, ch3, kernel_size=5, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.norm1 = nn.GroupNorm(1, ch1)
        self.norm2 = nn.GroupNorm(1, ch2)
        self.norm3 = nn.GroupNorm(1, ch3)

        self.fuse_conv = nn.Conv2d(mid_channel, output_channel, kernel_size=1, padding=0, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.begin(x)
        splits = torch.chunk(x1, 3, dim=1)
        s1, s2, s3 = splits

        y1 = self.norm1(self.branch1(s1))
        y2 = self.norm2(self.branch2(s2+y1))
        y3 = self.norm3(self.branch3(s3+y2))

        concat = torch.cat([y1, y2, y3], dim=1)
        out = x1 + concat
        out = self.fuse_conv(concat)
        out = self.act(out)
        return out
