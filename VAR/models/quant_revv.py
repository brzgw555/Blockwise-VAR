from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F
from torch_dct import dct_2d, idct_2d
from einops import rearrange, repeat

import dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]

def split_into_8x8_blocks(x):
    
    B, C, H, W = x.shape
    assert H % 8 == 0 and W % 8 == 0, "H,W must be divisible by 8"    

    # (B, C, H, W) → (B, C, num_blocks_h, 8, num_blocks_w, 8)
    blocks = rearrange(x, "b c (nh bh) (nw bw) -> b c nh nw bh bw", bh=8, bw=8)
    return blocks
def restore_from_8x8_blocks(blocks):
    
    B, C, num_blocks_h, num_blocks_w, h, w = blocks.shape
    assert h == 8 and w == 8, "patch_size 8x8"
    
    
    # (B, C, num_blocks_h, num_blocks_w, 8, 8) → (B, C, num_blocks_h, 8, num_blocks_w, 8)
    x = rearrange(blocks, "b c nh nw bh bw -> b c (nh bh) (nw bw)")
    return x
class EMAEmbedding(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        # init
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        # EMA accumulated cluster size
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        # EMA accumulated embedding
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        # update cluster
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        # update sum of features
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        
        n = self.cluster_size.sum()
        
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)
class VectorQuantizer2(nn.Module):
    
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 1.0,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4, dct_conv_layers=4,  # share_quant_resi: args.qsr
        ema_decay=0.9,ema_eps=1e-5
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        # ZIAN: Use loop for definition.
        self.scale=['1', '2', '4', '6', '8', '10', '13', '16']
        self.conv_params=[(2,2,0),(3,2,1),(3,2,1),(5,2,0),(3,2,1),(7,1,0),(4,1,0),(3,1,1)]  #(kernel_size,stride,padding)
        num_convs_map = {'1': 1, '2': 3, '4': 2, '6': 1, '8': 1, '10': 1, '13': 1, '16': 1}
        self.ema_update_step=8
        self.ema_step_num=0

        # ----------------- helpers -----------------
        def _conv3x3(in_ch, out_ch, bias=True):
            return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                            padding_mode='replicate', bias=bias)

        class AdaptiveScaleNorm(nn.Module):
            """
            Channel-wise LayerNorm-style conditioning for NCHW.
            Uses GroupNorm(1, C) as LN over channels per spatial location,
            then applies FiLM: y = norm(x) * (1 + gamma) + beta,
            where [gamma, beta] are predicted from a scale embedding.
            """
            def __init__(self, num_channels: int, emb_dim: int):
                super().__init__()
                self.norm = nn.GroupNorm(1, num_channels, eps=1e-6, affine=False)
                self.mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(emb_dim, 2 * num_channels)
                )

            def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
                """
                x: (N, C, H, W)
                emb: (N, D)  per-sample conditioning vector
                """
                h = self.mlp(emb)                       # (N, 2C)
                gamma, beta = h.chunk(2, dim=-1)        # (N, C), (N, C)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
                beta  = beta.unsqueeze(-1).unsqueeze(-1)   # (N, C, 1, 1)
                x = self.norm(x)
                return x * (1 + gamma) + beta

        # ----------------- BUILD -----------------
        emb_dim = Cvae  # small is fine; tune if you like

        # map scales to stable ids
        self.scale_to_idx = {str(s): i for i, s in enumerate(self.scale)}
        self.scale_embed  = nn.Embedding(len(self.scale), emb_dim)

        # --- ENCODER: shared convs ---
        enc_layers = []
        enc_layers += [_conv3x3(self.Cvae, self.Cvae), nn.SiLU(inplace=True)]
        for _ in range(max(0, dct_conv_layers - 1)):
            enc_layers += [_conv3x3(self.Cvae, self.Cvae), nn.SiLU(inplace=True)]
        self.enc_convs = nn.Sequential(*enc_layers)
        self.enc_adapter = AdaptiveScaleNorm(self.Cvae, emb_dim)
        self.enc_downsamples = nn.ModuleDict({
            str(s): (nn.Identity() if int(s) == 16 else nn.AdaptiveAvgPool2d((int(s), int(s))))
            for s in self.scale
        })

        # --- DECODER: shared upsample + shared convs ---
        self.dec_upsample = nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)
        self.dec_adapter = AdaptiveScaleNorm(self.Cvae, emb_dim)

        dec_layers = []
        for _ in range(max(0, dct_conv_layers - 1)):
            dec_layers += [_conv3x3(self.Cvae, self.Cvae), nn.SiLU(inplace=True)]
        dec_layers += [nn.Conv2d(self.Cvae, self.Cvae, kernel_size=1, bias=True)]
        self.dec_convs = nn.Sequential(*dec_layers)



        
        
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.register_buffer('epoch_ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = EMAEmbedding(self.vocab_size, self.Cvae, ema_decay ,ema_eps)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    def _scale_emb(self, s: str, batch: int, device):
        idx = torch.tensor(self.scale_to_idx[s], device=device, dtype=torch.long)
        e = self.scale_embed(idx)                # (emb_dim,)
        return e.unsqueeze(0).expand(batch, -1)  # (N, emb_dim)

    def get_codebook_usage(self):
        total_hits =self.ema_vocab_hit_SV.sum(dim=0)
        used_codes = (total_hits > 0).sum().item() 
        coverage = used_codes / self.vocab_size
        return coverage
    
    def reset_epoch_usage(self):
        self.epoch_ema_vocab_hit_SV.zero_()
        
    def get_epoch_codebook_usage(self):
        total_hits =self.epoch_ema_vocab_hit_SV.sum(dim=0)
        used_codes = (total_hits > 0).sum().item() 
        coverage = used_codes / self.vocab_size
        return coverage

    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        # f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_BChw)

        current_batch_cluster_size = torch.zeros(self.vocab_size, device=f_BChw.device)
        current_batch_embed_sum = torch.zeros(self.vocab_size, self.Cvae, device=f_BChw.device)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
                           
            f_split=split_into_8x8_blocks(f_BChw) #(B,C,H,W) -> (B,C, num_blocks_h, num_blocks_w,8, 8)
            # f_no_grad_split = split_into_8x8_blocks(f_no_grad) 
                
            f_split_dct= dct_2d(f_split, norm="ortho")
            
            for si, pn in enumerate(self.v_patch_nums): # from low to high
                dct_range= si+1
                scale = self.scale[si]

                f_dct_masked=torch.zeros_like(f_split)
                f_dct_masked[:,:,:,:,:dct_range,:dct_range]=f_split_dct[:,:,:,:,:dct_range,:dct_range]
                if si > 0:
                    f_dct_masked[:,:,:,:,:dct_range-1,:dct_range-1]=0
                f_dct_masked = idct_2d(f_dct_masked, norm="ortho")
                f_dct_masked = restore_from_8x8_blocks(f_dct_masked)
                #f_no_grad_dct_range = torch.zeros_like(f_no_grad_split,requires_grad=False)
                #f_no_grad_dct_range[:,:,:,:,:dct_range,:dct_range]=f_no_grad_split_dct[:,:,:,:,:dct_range,:dct_range]
                #f_no_grad_dct_range = idct_2d(f_no_grad_dct_range)
                #f_no_grad_dct_range = restore_from_8x8_blocks(f_no_grad_dct_range)
                
                # shared encoder
                emb = self._scale_emb(scale, f_dct_masked.size(0), f_dct_masked.device)            # (N, D)
                h   = self.enc_adapter(f_dct_masked, emb)                   # (N, C, 16, 16)
                h = self.enc_convs(h)                 # (N, C, 16, 16)

                downsample_f = self.enc_downsamples[scale](h)      # (N, C, s, s)


                downsample_f = rearrange(downsample_f, "b c h w -> (b h w) c")
                
                d_no_grad = torch.sum(downsample_f.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(downsample_f, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
                
                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)

                encodings = F.one_hot(idx_N, self.vocab_size).type(downsample_f.dtype)
                # feature sum：encodings.T @ downsample_f_flat → [vocab_size, Cvae]
                embed_sum = torch.matmul(encodings.T, downsample_f)

                if self.training and self.embedding.update:
                    current_batch_cluster_size += hit_V 
                    current_batch_embed_sum += embed_sum
                
                # calc loss
                downsample_f = downsample_f.reshape(B, pn ,pn ,C)
                idx_Bhw = idx_N.view(B, pn ,pn)
                h_BChw = self.embedding(idx_Bhw)
                mean_vq_loss += F.mse_loss(h_BChw,downsample_f.detach())
                mean_vq_loss += F.mse_loss(h_BChw.detach(),downsample_f).mul(self.beta)
                h_BChw = h_BChw + (downsample_f - downsample_f.detach())
                h_BChw = h_BChw.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)


                # h_BChw: (B, C, H, W)  with H=W=s for this scale
                h_BC16 = self.dec_upsample(h_BChw)                 # (B, C, 16, 16)
                emb = self._scale_emb(scale, h_BChw.size(0), h_BChw.device)              # (B, emb_dim)
                h_BC16 = self.dec_adapter(h_BC16, emb)             # (B, C, 16, 16)
                h_BChw = self.dec_convs(h_BC16)     
                

                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat = f_hat + h_BChw
                
                
                if self.training and dist.initialized():
                    handler.wait()
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                    self.epoch_ema_vocab_hit_SV[si].add_(hit_V)
                vocab_hit_V.add_(hit_V)
            
            if self.training and self.embedding.update:
            
                if dist.initialized():
                    tdist.all_reduce(current_batch_cluster_size)
                    tdist.all_reduce(current_batch_embed_sum)
            
                self.embedding.cluster_size_ema_update(current_batch_cluster_size)
                self.embedding.embed_avg_ema_update(current_batch_embed_sum)
                self.ema_step_num+=1
            
                self.embedding.weight_update(self.vocab_size)
                
                
                
            
            mean_vq_loss *=1. / SN



            
            # mean_vq_loss+=F.mse_loss(f_hat.detach(),f_BChw).mul_(self.beta)* 0.5
            # mean_vq_loss+= F.mse_loss(f_hat,f_no_grad)* 0.5
            
            
            # f_hat = (f_hat.detach() - f_no_grad).add(f_BChw)
        
        # margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        # if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        usages=self.get_codebook_usage()
        return f_hat, usages, mean_vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
