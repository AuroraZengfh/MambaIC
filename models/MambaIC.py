from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    GDN
)

from .VSS_module import conv, deconv, SS2D

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from datetime import datetime
import random
import time

from functools import partial
from typing import Optional, Callable, Any
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint

from einops.layers.torch import Rearrange


from timm.models.layers import trunc_normal_, DropPath, to_2tuple
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
import numpy as np
import math

global_var = 0
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )


class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x =  self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x


class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
        )
        
        self.drop_path = DropPath(drop_path)

    def _forward(self, input: torch.Tensor):
        if self.post_norm:
            x = input + self.drop_path(self.norm(self.op(input)))
        else:
            x = input + self.drop_path(self.op(self.norm(input)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            x = input.permute(0, 2, 3, 1)
            x = self._forward(x).permute(0, 3, 1, 2)
            return x
    
class MambaIC(CompressionModel):
    def __init__(self, depths=[2, 2, 9, 2], drop_path_rate=0.1, N=128,  M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.depths = depths
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.M = M
        
        assert num_slices == 5, 'implement number of context model to be 5 in the paper, otherwise you should modify the structure based on your demand.'

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        num_per_slice = M//self.num_slices
        
        self.m_down1 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[0])] + [conv(N*2, N*2, kernel_size=3, stride=2)]
        
        self.m_down2 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[1])] + [conv(N*2, N*2, kernel_size=3, stride=2)]
        
        self.m_down3 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[2])] + [conv(N*2, M, kernel_size=3, stride=2)]
        
        self.g_a = nn.Sequential(*[conv(3, N*2, kernel_size=5, stride=2)] + self.m_down1 + self.m_down2 + self.m_down3)

        self.ha_down1 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1] + depths[2]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[3])] + [conv(N*2, 192, kernel_size=3, stride=2)]
        
        self.h_a = nn.Sequential(
            *[conv(M, N*2, kernel_size=3, stride=2)] + self.ha_down1
        )

        depths = depths[::-1]

        self.hs_up1 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[0])] + [deconv(N*2, M, kernel_size=3, stride=2)]

        self.h_mean_s = nn.Sequential(
            *[deconv(192, N*2, kernel_size=3, stride=2)] + self.hs_up1
        )

        self.hs_up2 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[0])] + [deconv(N*2, M, kernel_size=3, stride=2)]

        self.h_scale_s = nn.Sequential(
            *[deconv(192, N*2, kernel_size=3, stride=2)] + self.hs_up2
        )

        self.m_up1 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[1])] + [deconv(N*2, N*2, kernel_size=3, stride=2)]
        
        self.m_up2 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[2])] + [deconv(N*2, N*2, kernel_size=3, stride=2)]
        
        self.m_up3 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1] + depths[2]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[3])] + [deconv(N*2, 3, kernel_size=5, stride=2)]

        self.g_s = nn.Sequential(*[deconv(M, N*2, kernel_size=3, stride=2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((M + num_per_slice*min(i, 5)), (M + num_per_slice*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((M + num_per_slice*min(i, 5)), (M + num_per_slice*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + num_per_slice*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, num_per_slice, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + num_per_slice*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, num_per_slice, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + num_per_slice*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, num_per_slice, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.anchor_atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten(num_per_slice, num_per_slice, 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.anchor_atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten(num_per_slice, num_per_slice, 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )


        c_depths = [2, 2, 2, 2, 2]
        context_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(c_depths))]
        self.context_vss_1 = nn.Sequential(*[VSSBlock(hidden_dim = 2*M + 2*num_per_slice, drop_path = context_dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(c_depths[0])],*[conv(2*M + 2*num_per_slice, 2*num_per_slice, kernel_size=3, stride=1)])
        
        self.context_vss_2 = nn.Sequential(*[VSSBlock(hidden_dim = 2*M + 4*num_per_slice,
                                 drop_path = context_dpr[i + c_depths[0]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                        for i in range(c_depths[1])], *[conv(2*M + 4*num_per_slice, 2*num_per_slice, kernel_size=3, stride=1)])
        
        self.context_vss_3 = nn.Sequential(*[VSSBlock(hidden_dim = 2*M + 4*num_per_slice,
                                 drop_path = context_dpr[i + c_depths[0] + c_depths[1]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                        for i in range(c_depths[2])], *[conv(2*M + 4*num_per_slice, 2*num_per_slice, kernel_size=3, stride=1)])
        
        self.context_vss_4 = nn.Sequential(*[VSSBlock(hidden_dim = 2*M + 4*num_per_slice,
                                 drop_path = context_dpr[i + c_depths[0] + c_depths[1] + c_depths[2]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                        for i in range(c_depths[3])], *[conv(2*M + 4*num_per_slice, 2*num_per_slice, kernel_size=3, stride=1)])
        
        self.context_vss_5 = nn.Sequential(*[VSSBlock(hidden_dim = 2*M + 4*num_per_slice,
                                 drop_path = context_dpr[i + c_depths[0] + c_depths[1] + c_depths[2] + c_depths[3]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                        for i in range(c_depths[4])], *[conv(2*M + 4*num_per_slice, 2*num_per_slice, kernel_size=3, stride=1)])
        
        self.context_vss = nn.ModuleList((self.context_vss_1, self.context_vss_2, self.context_vss_3, self.context_vss_4, self.context_vss_5))
        
        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
            num_per_slice, 2*num_per_slice, kernel_size=5, padding=2, stride=1
            ) for i in range(num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        return half * torch.erfc(const * inputs)

    def forward(self, x):
        y = self.g_a(x)
        B, C, H, W = y.size()
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        
        
        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = y.chunk(self.num_slices, 1)
        anchor_split = anchor.chunk(self.num_slices, 1)
        non_anchor_split = non_anchor.chunk(self.num_slices, 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * self.M//self.num_slices for _ in range(self.num_slices)], 1)
        
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [mu, scale, latent_means, latent_scales], dim=1)
            
            y_anchor = anchor_split[slice_index]
            
            means_anchor, scales_anchor = self.context_vss[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)


            means_anchor = self.anchor_atten_mean[slice_index](means_anchor)
            scales_anchor = self.anchor_atten_scale[slice_index](scales_anchor)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            y_anchor_quantilized = ste_round(y_anchor - means_anchor) + means_anchor
            y_anchor_quantilized_for_gs = ste_round(y_anchor - means_anchor) + means_anchor


            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0
            
            # checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, scales_non_anchor = self.context_vss[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            means_non_anchor = self.anchor_atten_mean[slice_index](means_non_anchor)
            scales_non_anchor = self.anchor_atten_scale[slice_index](scales_non_anchor)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]
            y_non_anchor_quantilized = ste_round(y_non_anchor - means_non_anchor) + means_non_anchor
            y_non_anchor_quantilized_for_gs = ste_round(y_non_anchor - means_non_anchor) + means_non_anchor

            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
                    
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slice_for_gs += lrp
            
            y_hat_slices.append(y_hat_slice)
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)
            
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y": y,
            "z": z
        }
        
    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        B, C, H, W = y.size()

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        y_slices = y.chunk(self.num_slices, 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * self.M//self.num_slices for _ in range(self.num_slices)], 1)
        
        y_hat_slices = []
        y_strings = []
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [mu, scale, latent_means, latent_scales], dim=1)

            y_anchor = y_slices[slice_index].clone()
            
            means_anchor, scales_anchor = self.context_vss[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            means_anchor = self.anchor_atten_mean[slice_index](means_anchor)
            scales_anchor = self.anchor_atten_scale[slice_index](scales_anchor)
            
            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()
            y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)
            
            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode)
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode)
            
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.context_vss[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            means_non_anchor = self.anchor_atten_mean[slice_index](means_non_anchor)
            scales_non_anchor = self.anchor_atten_scale[slice_index](scales_non_anchor)

            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode)
            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_hat_slice = y_anchor_decode + y_non_anchor_quantized
                    
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            
            y_hat_slices.append(y_hat_slice)
            y_strings.append([anchor_strings, non_anchor_strings])

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B, _, _, _ = z_hat.size()

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_strings = strings[0]

        ctx_params_anchor_split = torch.split(torch.zeros(B, self.M * 2, z_hat.shape[2] * 4, z_hat.shape[3] * 4).to(z_hat.device),
                                              [2 * self.M//self.num_slices for _ in range(self.num_slices)], 1)
        
        y_hat_slices = []
        
        for slice_index in range(self.num_slices):
            
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [mu, scale, latent_means, latent_scales], dim=1)        
            
            means_anchor, scales_anchor = self.context_vss[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)


            means_anchor = self.anchor_atten_mean[slice_index](means_anchor)
            scales_anchor = self.anchor_atten_scale[slice_index](scales_anchor)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]
            
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                    means=means_anchor_encode)


            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]
            
            # checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.context_vss[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            means_non_anchor = self.anchor_atten_mean[slice_index](means_non_anchor)
            scales_non_anchor = self.anchor_atten_scale[slice_index](scales_non_anchor)

            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]
            
            # entropy estimation
            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)

            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
                    
            lrp_support = torch.cat([mean_support, y_slice_hat], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_slice_hat += lrp
            
            y_hat_slices.append(y_slice_hat)
            
        y_hat = torch.cat(y_hat_slices, dim=1)

        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}