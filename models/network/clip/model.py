import math
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from options.options import Options
from util import read_yaml

# additional code -----------------------------------------
options = Options()
opt = options.parse()

class WSGM(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, dropout_prob=0.5):
        super(WSGM, self).__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.relu1 = nn.ReLU()
        self.middle = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.relu2 = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.dropout1 = nn.Dropout(dropout_prob)  # Independent Dropout layers
        self.dropout2 = nn.Dropout(dropout_prob)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.down.bias, 0)
        nn.init.normal_(self.middle.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.middle.bias, 0)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.up.bias, 0)

    def forward(self, x):
        x = self.down(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.middle(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.up(x)
        return x


# original CLIP code -----------------------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_wsgm: bool = False, wsgm_module: nn.Module = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_wsgm = use_wsgm
        self.wsgm_module = wsgm_module
        # SVD 正交约束：语义投影矩阵 (作为普通属性，不需要梯度)
        self.semantic_proj_matrix = None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # New: Manual attention computation for RCS token extraction
    def manual_attn(self, x, return_map=False):
        # x: [L, B, D] -> [Seq_len, Batch, Dim]
        num_tokens, bsz, embed_dim = x.size()
        num_heads = self.attn.num_heads
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        # Extract Q, K, V from in_proj_weight and in_proj_bias
        if self.attn.in_proj_bias is None:
            # Fallback for older PyTorch versions
            q = F.linear(x, self.attn.q_proj_weight, None)
            k = F.linear(x, self.attn.k_proj_weight, None)
            v = F.linear(x, self.attn.v_proj_weight, None)
        else:
            q, k, v = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, dim=-1)

        # Transform dimensions to [B, Num_heads, Seq_len, Head_dim]
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # Compute Attention Map [B*Num_heads, Seq_len, Seq_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        if return_map:
            return attn_weights, v

        # Standard output computation
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_tokens, bsz, embed_dim)
        attn_output = self.attn.out_proj(attn_output)
        return attn_output

    def forward(self, x: torch.Tensor, return_orth_loss: bool = False):
        x = x + self.attention(self.ln_1(x))

        orth_loss = 0.0
        if self.use_wsgm and self.wsgm_module is not None:
            wsgm_output = self.wsgm_module(x)
            x = x + wsgm_output

            # 计算正交损失：WSGM 输出在语义子空间的投影
            if return_orth_loss and self.semantic_proj_matrix is not None:
                # 将 wsgm_output 投影到语义子空间
                # wsgm_output: [seq_len, batch, d_model]
                # semantic_proj_matrix: [d_model, rank]
                # 使用 torch.matmul 处理矩阵乘法
                wsgm_flat = wsgm_output.permute(1, 0, 2).reshape(-1, wsgm_output.size(-1))  # [batch*seq_len, d_model]
                # 确保 semantic_proj_matrix 在正确的设备上
                proj_matrix = self.semantic_proj_matrix.to(wsgm_flat.device)
                proj = torch.matmul(wsgm_flat, proj_matrix)  # [batch*seq_len, rank]
                # detach 确保梯度只从 orth_loss 回传，不影响主图
                orth_loss = torch.mean(proj ** 2).detach()

        x = x + self.mlp(self.ln_2(x))

        if return_orth_loss:
            return x, orth_loss
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_wsgm_modules: int = 8,
                 scale: float = 1.0, svd_rank: int = 64, svd_energy_ratio: float = 0.9):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_wsgm_modules = min(num_wsgm_modules, layers)
        self.scale = scale
        self.svd_rank = svd_rank
        self.svd_energy_ratio = svd_energy_ratio

        self.wsgm_out_dim = int(self.scale * (self.width // opt.WSGM_reduction_factor))

        self.WSGM_modules = nn.ModuleList([
            WSGM(width, self.wsgm_out_dim)
            for _ in range(num_wsgm_modules)
        ])

        # 先创建 resblocks，再计算 SVD
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width,
                heads,
                attn_mask,
                use_wsgm=True,
                wsgm_module=self._select_wsgm_module(i)
            )
            for i in range(layers)
        ])

        # SVD 正交约束：计算语义投影矩阵
        self.semantic_proj_matrix = self._compute_semantic_projection_matrix()

        # 将语义投影矩阵注册到每个 resblock
        if self.semantic_proj_matrix is not None:
            for block in self.resblocks:
                block.semantic_proj_matrix = self.semantic_proj_matrix

    def _compute_semantic_projection_matrix(self) -> torch.Tensor:
        """
        计算语义投影矩阵 (Semantic Projection Matrix)

        基于 Effort 论文思想：
        - 对冻结的 CLIP Attention 权重进行 SVD 分解
        - 提取前 K 个主成分作为语义子空间基向量
        - WSGM 应尽量避免在这个子空间学习（保持正交）

        Returns:
            semantic_proj_matrix: [d_model, rank] 投影矩阵
        """
        # 收集所有层的 in_proj_weight 进行 SVD
        all_weights = []
        for block in self.resblocks:
            if hasattr(block.attn, 'in_proj_weight') and block.attn.in_proj_weight is not None:
                # in_proj_weight 形状为 [d_model * 3, d_model]，包含 Q, K, V 投影
                all_weights.append(block.attn.in_proj_weight)

        if len(all_weights) == 0:
            # 如果没有 in_proj_weight，使用 q_proj_weight
            for block in self.resblocks:
                if hasattr(block.attn, 'q_proj_weight') and block.attn.q_proj_weight is not None:
                    all_weights.append(block.attn.q_proj_weight)

        if len(all_weights) == 0:
            # 默认返回 None（不应用正交约束）
            return None

        # 拼接所有权重矩阵
        combined_weights = torch.cat(all_weights, dim=0)  # [N * d_model, d_model]

        # SVD 分解: combined_weights = U @ S @ V^T
        # 我们使用 V 的列作为主成分方向
        try:
            # 使用经济型 SVD，只计算需要的部分
            U, S, Vh = torch.linalg.svd(combined_weights, full_matrices=False)
            V = Vh.T  # V 的形状与 combined_weights 相同

            # 计算累积能量，确定保留的主成分数量
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)

            # 方案1：按能量比例保留
            rank_by_energy = torch.searchsorted(cumulative_energy, self.svd_energy_ratio) + 1
            # 方案2：按指定 rank 保留
            rank = min(self.svd_rank, rank_by_energy, V.size(1))

            # 取前 rank 个奇异向量作为语义基
            semantic_basis = V[:, :rank]  # [d_model, rank]

            print(f"[SVD Orthogonal Constraint] Semantic subspace rank: {rank} "
                  f"(energy_ratio={self.svd_energy_ratio}, svd_rank={self.svd_rank})")

            return semantic_basis.float()

        except Exception as e:
            print(f"[SVD Warning] SVD computation failed: {e}, disabling orth constraint")
            return None

    def _select_wsgm_module(self, layer_idx: int):
        wsgm_idx = (layer_idx * self.num_wsgm_modules) // self.layers
        return self.WSGM_modules[wsgm_idx]

    def forward(self, x: torch.Tensor, return_rcs=False, return_orth_loss: bool = False):
        out = {}
        intermediate_attns = []
        rcs_token = None
        total_orth_loss = 0.0

        # Define RCS layers (5-9 as suggested by ResCLIP)
        rcs_layers = range(5, min(9, self.layers))

        x_input = x
        for idx, layer in enumerate(self.resblocks.children()):
            # Extract Attention Map if it's an RCS layer
            if return_rcs and idx in rcs_layers:
                with torch.no_grad():
                    # Get normalized input
                    normalized_input = layer.ln_1(x_input)
                    attn_map, _ = layer.manual_attn(normalized_input, return_map=True)
                    intermediate_attns.append(attn_map)

            # If it's the last layer, we need its Value to generate RCS Token
            if return_rcs and idx == self.layers - 1:
                with torch.no_grad():
                    normalized_input = layer.ln_1(x_input)
                    _, v_last = layer.manual_attn(normalized_input, return_map=True)

            # Normal layer computation (支持返回正交损失)
            if return_orth_loss:
                x_input, orth_loss = layer(x_input, return_orth_loss=True)
                total_orth_loss = total_orth_loss + orth_loss
            else:
                x_input = layer(x_input)

            out[f'layer{idx}'] = x_input[0]

        # Compute RCS Token if requested
        if return_rcs and len(intermediate_attns) > 0:
            # Aggregate intermediate layer Attention Maps
            # Shape: [Batch*Heads, Seq_len, Seq_len]
            avg_attn = torch.stack(intermediate_attns).mean(dim=0)

            # We only care about CLS token's attention to other patches (Row 0)
            # avg_attn shape: [B*H, N+1, N+1], take [:, 0:1, :] -> CLS row
            rcs_attn_cls = avg_attn[:, 0:1, :]

            # Use RCS Attention to aggregate last layer's Value
            # v_last shape: [B*H, N+1, Head_Dim]
            # rcs_head_out = rcs_attn_cls @ v_last -> [B*H, 1, Head_Dim]
            rcs_head_out = torch.bmm(rcs_attn_cls, v_last)

            # Restore dimensions properly
            # rcs_head_out: [B*H, 1, Head_Dim]
            # We need to concatenate across heads to get [B, 1, D] where D = H * Head_Dim
            bsz = x.shape[1]  # Note: x is [Seq, Batch, Dim]
            num_heads = self.resblocks[0].attn.num_heads
            head_dim = self.width // num_heads

            # Reshape from [B*H, 1, Head_Dim] to [B, H, Head_Dim]
            rcs_token = rcs_head_out.view(bsz, num_heads, head_dim)
            # Concatenate across heads: [B, H*Head_Dim] = [B, width]
            rcs_token = rcs_token.reshape(bsz, 1, self.width)
            # Transpose to [1, B, width]
            rcs_token = rcs_token.transpose(0, 1)

        if return_orth_loss:
            return out, x_input, rcs_token, total_orth_loss
        return out, x_input, rcs_token
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # SVD 正交约束参数
        svd_rank = getattr(opt, 'svd_rank', 64)
        svd_energy_ratio = getattr(opt, 'svd_energy_ratio', 0.9)

        self.transformer = Transformer(
            width, layers, heads,
            num_wsgm_modules=opt.WSGM_count,
            svd_rank=svd_rank,
            svd_energy_ratio=svd_energy_ratio
        )
        # self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_rcs=False, return_orth_loss: bool = False):
        x = self.conv1(x)  # shape = [batch_size, patch_size, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, patch_size, patch_num]
        x = x.permute(0, 2, 1)  # shape = [*, patch_num, patch_size]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, patch_num + 1, patch_size]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if return_orth_loss:
            out, x, rcs_token, orth_loss = self.transformer(x, return_rcs=return_rcs, return_orth_loss=True)
        else:
            out, x, rcs_token = self.transformer(x, return_rcs=return_rcs)
            orth_loss = None
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        cls_tokens = [self.ln_post(out['layer' + str(idx)]) @ self.proj for idx in range(len(out))]

        out['before_projection'] = x

        if self.proj is not None:
            x = x @ self.proj
        out['after_projection'] = x

        # Return CLIP features, cls_tokens, and optionally RCS token
        if return_rcs:
            # Project RCS token to match cls_tokens dimensions
            if rcs_token is not None and self.proj is not None:
                rcs_token = rcs_token @ self.proj  # [1, B, 768] -> [1, B, 1024]
            if return_orth_loss:
                return x, cls_tokens, rcs_token, orth_loss
            return x, cls_tokens, rcs_token
        else:
            if return_orth_loss:
                return x, cls_tokens, orth_loss
            return x, cls_tokens


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_rcs=False, return_orth_loss: bool = False):
        return self.visual(image.type(self.dtype), return_rcs=return_rcs, return_orth_loss=return_orth_loss)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    # model.load_state_dict(state_dict)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model.eval()
