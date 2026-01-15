# 输入图像 -> 初始卷积(head) -> 下采样路径(downblocks) -> KAN处理第一阶段(patch_embed3 + kan_block1)
#            -> KAN处理第二阶段(patch_embed4 + kan_block2) -> KAN解码第二阶段(decoder1 + kan_dblock1)
#            -> KAN解码第一阶段(decoder2) -> 上采样路径(upblocks) -> 输出卷积(tail) -> 输出图像
   
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, to_3tuple, trunc_normal_
from einops import rearrange

# 结合了传统线性变换和 B 样条插值的混合层
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,                          # 网格大小
        spline_order=3,                       # 样条阶数，spline_order: 样条阶数(默认为3，即立方样条)
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 创建网格 grid: 注册为buffer的样条网格，形状为(in_features, grid_size + 2*spline_order + 1)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # 可训练参数 base_weight: 传统线性变换的权重矩阵；spline_weight: 样条系数权重，形状为(out_features, in_features, grid_size + spline_order)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # 可选的独立样条缩放参数。spline_scaler: 可选的独立样条缩放参数
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise              # 噪声缩放
        self.scale_base = scale_base                # 基础缩放
        self.scale_spline = scale_spline            # 样条缩放
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()    # 基础激活函数
        self.grid_eps = grid_eps                    # 网格更新混合系数

        # 初始化参数
        self.reset_parameters()

    # 使用Kaiming均匀初始化基础权重，初始化样条权重为随机噪声并通过curve2coeff转换为样条系数。curve2coeff函数将样条系数转换为样条曲线的功能主要是为了根据给定的样条系数计算样条曲线的实际值。
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)         # 使用Kaiming均匀初始化基础权重
        with torch.no_grad():               # 创建随机噪声
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            # 计算样条系数
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            # 初始化样条缩放参数（如果启用）
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    # B 样条计算
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)            # 初始化基函数（矩形函数）

        # 递归计算B样条基
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    #曲线到系数转换
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # # 计算样条基
        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)

        # 解最小二乘问题
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        # 基础线性变换 + 激活函数  
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 样条变换
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output                  #两者相加得到输出

    @torch.no_grad()

    # 网格更新 update_grid
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # 计算当前样条输出
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        # 自适应网格（基于输入分布）
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 混合两种网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以包含样条阶数范围
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新网格和样条权重
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        # 计算L1伪正则化（样条权重的平均绝对值）
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()

        # 熵正则化
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

# 继承自 torch.nn.Module。它是一个多层网络，由多个 KANLinear 层组成，支持基于样条（spline）的激活函数和动态网格更新。
class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        #构建网络层：使用 torch.nn.ModuleList() 存储所有层。
        self.layers = torch.nn.ModuleList()

        #遍历 layers_hidden 的相邻元素（如 (input_dim, hidden1), (hidden1, hidden2)），为每对输入/输出维度创建一个 KANLinear 层，并传入初始化参数。
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

# 重叠分块嵌入模块，用于将输入图像分割成多个重叠的图像块（patches），并通过卷积投影到高维嵌入空间。
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        if stride is None: stride = patch_size    # By default, non-overlapping patches
        stride = (1, stride, stride)

        self.img_size = img_size
        self.patch_size = patch_size
       # self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
       # self.num_patches = self.H * self.W     # 总块数

        # 核心卷积投影层：用卷积代替非重叠分块
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=1)
        
        self.norm = nn.LayerNorm(embed_dim)         # 对嵌入向量归一化

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)                  # Now shape: [B, embed_dim, t', h', w']
        B, C, T, H, W = x.shape              # 提取投影后的高和宽
        x = rearrange(x, 'b c t h w -> (b t) c h w')        # 展平并转置：[(B*T), embed_dim, H, W]

        x = x.flatten(2).transpose(1, 2)       # 展平并转置：[(B*T),  (H*W),  embed_dim]
        x = self.norm(x)                   # 归一化

        return x, H, W, T


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
def swish(x):
    
    return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

#实现一个基于 KANLinear的特征变换模块    
class kan(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        # KANLinear 的超参数
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=Swish
        grid_eps=0.02
        grid_range=[-1, 1]

        # 核心：KANLinear 层
        self.fc1 = KANLinear(
                    in_features,
                    hidden_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):

        B, N, C = x.shape                    # 输入：维度为：[(B*T), (H*W), C]
        x = self.fc1(x.reshape(B*N,C))      # 展平为 [B*N, C] 输入 KANLinear
        x = x.reshape(B,N,C).contiguous()    # 恢复形状

        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4.,drop_path=0.,norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)             # 扩展维度

        # 时间嵌入投影（256维 -> dim）
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(1280, dim),     # 1280->1280
        )

        # 核心：kan 模块
        self.kan = kan(in_features=dim, hidden_features=mlp_hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, temb):
        # x 的shape： [(B*T),  (H*W),  embed_dim]
        temb = self.temb_proj(temb)      # 投影时间嵌入

        x = self.drop_path(self.kan(self.norm2(x), H, W))        # norm2：归一化，kan 变换，drop_path(x)：随机深度
        N = int(x.shape[0]/temb.shape[0])
        temb = temb.unsqueeze(1).repeat(N, 1, 1)
        #print("-------------shiftedBlock_N:", N)
        #print("-------------shiftedBlock_x,temb:", x.shape, temb.shape)

        x = x + temb                  #残差连接 + 时间条件 x: [(B*T),  (H*W),  embed_dim], temb: [B, 1, C]

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).view(B * T * N, C, H, W)
        x = self.dwconv(x)
        x = x.view(B, T, N, C).contiguous()

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.GroupNorm(32, dim)
        # self.relu = Swish()

    def forward(self, x, H, W):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).view(B * T * N, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = swish(x)
        x = x.view(B, T, N, C).contiguous()

        return x

# 简单的卷积块，支持时间嵌入的条件融合。
class SingleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
        )

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(1280, h_ch),                        # temb 输入维度
        )
    def forward(self, input, temb):
        B, T, C, H, W = input.shape
        input = input.view(B * T, C, H, W)
        x = self.conv(input)
        x = x.view(B, T, -1, H, W)
        temb = self.temb_proj(temb)
        return x + temb.unsqueeze(1).unsqueeze(1)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Conv2d(h_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish()
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input, temb):
        B, C, T, H, W = input.shape
        input = input.view(B * T, C, H, W)
        x = self.conv(input)
        x = x.view(B, T, -1, H, W)
        temb = self.temb_proj(temb)
        return x + temb.unsqueeze(1).unsqueeze(1)

class D_SingleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_SingleConv, self).__init__()
        #nn.Sequential()按顺序组合多个神经网络层
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv3d(in_ch, h_ch, (3, 3, 3), (1, 1, 1), padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(1280, h_ch),
        )
    def forward(self, input, temb):
        B, C, T, H, W = input.shape

        x = self.conv(input)

        temb = self.temb_proj(temb)
        temb = temb[:, :, None, None, None]  # 调整为(B, h_ch, 1, 1, 1)
        return x + temb

#两层卷积的堆叠，类似 U-Net 的基本块。
class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(32,in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
             nn.GroupNorm(32,h_ch),
            Swish()
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input,temb):
        B, T, C, H, W = input.shape
        input = input.view(B * T, C, H, W)
        x = self.conv(input)
        x = x.view(B, T, -1, H, W)
        temb = self.temb_proj(temb)
        return x + temb.unsqueeze(1).unsqueeze(1)

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.view(B * T, C, -1).permute(0, 2, 1)
        k = k.view(B * T, C, -1)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B*T, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B*T, H * W, C]
        h = h.view(B*T, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, h_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, h_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(h_ch, h_ch, 3, stride=1, padding=1),
        )
        if in_ch != h_ch:
            self.shortcut = nn.Conv2d(in_ch, h_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(h_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        h = self.block1(x)
        temb = self.temb_proj(temb)
        h = h + temb.unsqueeze(2).unsqueeze(3)
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        h = h.view(B, T, -1, H, W)
        return h


class UKan_Hybrid(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):    
        # T: 时间步数（可能用于扩散模型）； 
        # ch: 基础通道数； 
        # ch_mult: 通道数乘数列表，决定每层的通道数； 
        # attn: 指定哪些层使用注意力机制； 
        # num_res_blocks: 每个下采样/上采样阶段的残差块数量；   
        # dropout: dropout概率
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index h of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)            #创建时间嵌入模块，将时间步t编码为向量
        attn = []
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record hput channel when dowmsample for upsample
        now_ch = ch

        # 循环构建下采样路径： 
        for i, mult in enumerate(ch_mult):
            h_ch = ch * mult                     # 1. 每个阶段开始时根据ch_mult增加通道数;
            for _ in range(num_res_blocks):
                # 2. 添加num_res_blocks个残差块，某些阶段会添加注意力;
                self.downblocks.append(ResBlock(        # downblocks: 下采样模块列表
                    in_ch=now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:                         # 3. 除非是最后一个阶段，否则添加下采样层
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)                               # chs列表记录每层的输出通道数，供上采样路径使用

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            h_ch = ch * mult
            for _ in range(num_res_blocks + 1):                #upblocks: 上采样模块列表每个阶段：使用num_res_blocks + 1个残差块（比下采样多一个）;
                self.upblocks.append(ResBlock(                          # 1. 输入通道是跳跃连接的特征+当前特征; 
                    in_ch=chs.pop() + now_ch, h_ch=h_ch, tdim=tdim,           # 2. 输入通道是跳跃连接的特征+当前特征;
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
            if i != 0:                                           # 3. 除非是第一个阶段，否则添加上采样层
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        # 输出层： 归一化+激活函数+1x1卷积，将特征映射回3通道输出
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )

      
        # --KAN 相关模块。 这部分实现了：两个重叠补丁嵌入层 (patch_embed3, patch_embed4)。三个归一化层。三个KAN块（两个用于编码，一个用于解码）。两个解码器模块，用于上采样特征。
        # KAN 相关参数
        embed_dims = [256, 320, 512]
        norm_layer = nn.LayerNorm
        dpr = [0.0, 0.0, 0.0]

        # 补丁嵌入层
        self.patch_embed3 = OverlapPatchEmbed(img_size=64 // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=64 // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])


        # 归一化层
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])

        # KAN 块
        self.kan_block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1],  mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer)])

        self.kan_block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2],  mlp_ratio=1, drop_path=dpr[1], norm_layer=norm_layer)])

        self.kan_dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer)])

        # 解码器
        self.decoder1 = D_SingleConv(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_SingleConv(embed_dims[1], embed_dims[0])  

        # --KAN 相关模块。



        self.initialize()

    #初始化方法。 对头卷积和尾卷积进行Xavier初始化。尾卷积使用较小的增益值(1e-5)，可能是为了稳定训练。
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    # 前向传播流程：   
    # 1. 时间嵌入：将时间步t编码为向量temb   
    # 2. 初始卷积：通过head模块处理输入图像    
    # 3. 下采样路径： 保存每层特征到hs列表（用于跳跃连接）； 最终特征保存为t3
    # 4. KAN处理： 第一阶段：通过patch_embed3和kan_block1； 第二阶段：进一步下采样并通过kan_block2
    # 5. KAN解码： 第二阶段解码：上采样并与t4残差连接； 第一阶段解码：进一步上采样并与t3残差连接
    # 6. 上采样路径： 使用跳跃连接（concat特征）； 通过upblocks模块逐步上采样。
    # 7. 输出：通过tail模块生成最终输出

    #架构特点：    
    #  混合结构：结合了U-Net的跳跃连接和KAN的注意力机制；   
    #  多尺度处理：既有传统的CNN下采样/上采样路径，又有基于补丁的Transformer风格处理；   
    #  残差连接：在KAN解码部分使用了残差连接；   
    #  时间嵌入：支持时间步输入，适合扩散模型等任务；   
    #  灵活的注意力配置：可以通过attn参数控制哪些层使用注意力

    def forward(self, x, t):         #前向过程
        # Timestep embedding  # 时间步嵌入
        temb = self.time_embedding(t)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # Downsampling              # 下采样路径
        h = self.head(x)             # head

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        #------------------------------------------------
        t3 = h        # 保存下采样最终特征

        # KAN 处理第一阶段
        B = x.shape[0]
        h, H, W = self.patch_embed3(h)              # h的维度为：[B, (T*H*W), embed_dim]
 
        for i, blk in enumerate(self.kan_block1):
            h = blk(h, H, W, temb)
        h = self.norm3(h)                        #[B,N,C]            
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = h            # 保存第一阶段KAN特征

        # KAN 处理第二阶段
        h, H, W= self.patch_embed4(h)
        for i, blk in enumerate(self.kan_block2):
            h = blk(h, H, W, temb)
        h = self.norm4(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        # 解码第二阶段
        h = swish(F.interpolate(self.decoder1(h, temb), scale_factor=(2,2), mode ='bilinear'))

        h = torch.add(h, t4)       # 残差连接

        # 返回第一阶段KAN格式
        _, _, T, H, W = h.shape
        h = h.flatten(2).transpose(1,2)               #[B,N,C]
        for i, blk in enumerate(self.kan_dblock1):
            h = blk(h, H, W, temb)

            
        ### Stage 3 # 解码第一阶段
        h = self.dnorm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        h = swish(F.interpolate(self.decoder2(h, temb),scale_factor=(2,2),mode ='bilinear'))

        h = torch.add(h,t3)         # 残差连接
        #------------------------------------------------

        # Upsampling         # 上采样路径
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)          # 跳跃连接
            h = layer(h, temb)
        
        h = self.tail(h)                      # tail

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UKan_Hybrid(
        T=1000, ch=64, ch_mult=[1, 2, 2, 2], attn=[],
        num_res_blocks=2, dropout=0.1)

