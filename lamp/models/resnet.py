# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .KAN import KANLinear  # Wav-KAN

class InflatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_temp=True):
        super().__init__()
        self.use_temp = True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.stride = stride
        self.padding = padding
        if isinstance(padding, tuple):
            padding = padding[0]

        self.conv_gate = nn.Conv2d(out_channels, 1, 3, stride=1, padding=1)
        nn.init.constant_(self.conv_gate.weight, 0)
        nn.init.constant_(self.conv_gate.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.conv1d = nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=2)
    
    def forward(self, x):
        video_length, h, w = x.shape[-3:]
        x_2d = rearrange(x, "b c f h w -> (b f) c h w")
        x_2d = self.conv2d(x_2d)
        x_2d = rearrange(x_2d, "(b f) c h w -> b c f h w", f=video_length)
        if self.use_temp:
            x_gate = rearrange(x_2d, "b c f h w -> (b f) c h w")
            c = x_gate.shape[1]
            x_gate = self.sigmoid(self.conv_gate(x_gate)).repeat(1, c, 1, 1)
            x_gate = rearrange(x_gate, "(b f) c h w -> b c f h w", f=video_length)

            x_1d = rearrange(x_2d, "b c f h w -> (b h w) c f", f=video_length)
            x_1d = self.conv1d(x_1d)[:, :, :-2]
            h, w = x_2d.shape[-2:]
            x_1d = rearrange(x_1d, "(b h w) c f -> b c f h w", h=h, w=w)
            x = x_1d * x_gate + x_2d
        else:
            x = x_2d

        return x
    
# WavKANInflatedConv3d, replace Conv1d layer in InflatedConv3d, retain: Conv2d + Conv_gate
class WavKANInflatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 use_temp=True, wavelet_type='morlet', kan_hidden_dim=None):
        super().__init__()
        self.use_temp = use_temp
        self.wavelet_type = wavelet_type
        self.out_channels = out_channels

        # --- space branch (retain) ---
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # --- gate (retain) ---
        self.conv_gate = nn.Conv2d(out_channels, 1, 3, stride=1, padding=1)
        nn.init.constant_(self.conv_gate.weight, 0)
        nn.init.constant_(self.conv_gate.bias, 0)
        self.sigmoid = nn.Sigmoid()

        # --- new: Wav-KAN time branch (replace Conv1d) ---
        if self.use_temp:
            # KANLinear: It is defined as transforming the feature vector at each time step.
            # kan_hidden_dim is used to build deeper KAN layers (optional)
            if kan_hidden_dim is None:
                # simple: single layer KANLinear
                self.temporal_kan = KANLinear(out_channels, out_channels, wavelet_type=wavelet_type)
            else:
                # if multi-layer KANLinear, use nn.Sequential
                # Note:KANLinear is embedded with batch normalization, which may require parameter adjustment during stacking.
                self.temporal_kan = nn.Sequential(
                    KANLinear(out_channels, kan_hidden_dim, wavelet_type=wavelet_type),
                    KANLinear(kan_hidden_dim, out_channels, wavelet_type=wavelet_type)
                )

    def forward(self, x):
        # x shape: (B, C, F, H, W)
        video_length = x.shape[2]
        h, w = x.shape[-2:]

        # --- space branch ---
        x_2d = rearrange(x, "b c f h w -> (b f) c h w")
        x_2d = self.conv2d(x_2d)
        x_2d = rearrange(x_2d, "(b f) c h w -> b c f h w", f=video_length)

        if self.use_temp:
            # --- gate ---
            x_gate = rearrange(x_2d, "b c f h w -> (b f) c h w")
            c_gate = x_gate.shape[1]
            gate = self.sigmoid(self.conv_gate(x_gate))
            gate = gate.repeat(1, c_gate, 1, 1)  # expand the channel dimension
            gate = rearrange(gate, "(b f) c h w -> b c f h w", f=video_length)

            # --- time branch :  Wav-KAN replace Conv1d ---
            # 1. Reshape x2d​ into (B∗H∗W,C,F) to independently process the time series at each spatial position.
            B, C, F, H, W = x_2d.shape
            x_1d = rearrange(x_2d, "b c f h w -> (b h w) f c")  # (B*H*W, F, C)

            # 2.  Wav-KAN
            # KANLinear input shape : (N, in_features), process each time step separately.
            # method:  x_1d reshape (B*H*W*F, C)
            N, F_len, C_in = x_1d.shape
            x_flat = x_1d.reshape(-1, C_in)  # (B*H*W*F, C)
            
            x_flat = self.temporal_kan(x_flat)  # (B*H*W*F, out_channels)
            
            x_1d = x_flat.reshape(N, F_len, self.out_channels)  # (B*H*W, F, out_channels)
            x_1d = rearrange(x_1d, "(b h w) f c -> b c f h w", b=B, h=H, w=W, f=F_len)

            # 3. gate fusion
            x = x_1d * gate + x_2d
        else:
            x = x_2d

        return x


class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv", use_temp=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1, use_temp=use_temp)
            #conv = WavKANInflatedConv3d(self.channels, self.out_channels, 3, padding=1, use_temp=use_temp, wavelet_type='morlet', kan_hidden_dim=None)  #'mexican_hat'

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv", use_temp=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding, use_temp=use_temp)
            #conv = WavKANInflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding, use_temp=use_temp, wavelet_type='morlet', kan_hidden_dim=None)

        else:
            raise NotImplementedError

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        use_temp=True,
        is_mid_block=False
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        #self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_temp=use_temp)
        #self.conv1 = WavKANInflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_temp=use_temp, wavelet_type='morlet', kan_hidden_dim=None)
        # ===================== use Wav-KAN only in Bottleneck layer  =====================
        if is_mid_block:
            self.conv1 = WavKANInflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_temp=use_temp, wavelet_type='morlet', kan_hidden_dim=None)
        else:
            self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_temp=use_temp)
        # ======================================================================

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels           # add feature maps
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2        # Scaling + shifting
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)

        #self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_temp=use_temp)
        #self.conv2 = WavKANInflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_temp=use_temp, wavelet_type='morlet', kan_hidden_dim=None)
        # ===================== only in bottleneck layer=====================
        if is_mid_block:
            self.conv2 = WavKANInflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_temp=use_temp, wavelet_type='morlet', kan_hidden_dim=None)
        else:
            self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_temp=use_temp)
        # ======================================================================

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            #self.conv_shortcut = WavKANInflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, wavelet_type='morlet', kan_hidden_dim=None)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        # =========== Block A: Preprocessing ===========
        hidden_states = self.norm1(hidden_states)             # [B,C,T,H,W] → GroupNorm
        hidden_states = self.nonlinearity(hidden_states)         # SiLU.

        hidden_states = self.conv1(hidden_states)             # InflatedConv3d


        # =========== Block B: Time Conditioning ===========
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]       # Broadcast time vectors to the identical spatial dimension

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb                                      

        hidden_states = self.norm2(hidden_states)                             # GroupNorm

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)                       
            hidden_states = hidden_states * (1 + scale) + shift              # FiLM

        # =========== Block C: Postprocessing ===========
        hidden_states = self.nonlinearity(hidden_states)                  

        hidden_states = self.dropout(hidden_states)                      # Dropout
        hidden_states = self.conv2(hidden_states)                       # InflatedConv3d           

        # =========== Block D: Shortcut & Fusion ===========
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor          

        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))