# ForgeLens SVD 正交约束修改总结

## 📋 概述

本修改将 **Effort 论文** 的 **SVD 正交子空间分解** 思想引入 ForgeLens 的 Stage 1 训练，通过强制 WSGM 模块学习与 CLIP 语义特征正交的伪造痕迹，提升检测性能。

**修改时间**: 2026年1月
**核心思想**: 将特征空间分解为语义子空间（Semantic Subspace）和伪造子空间（Forgery Subspace），迫使 WSGM 在伪造子空间中学习。

---

## 🏗️ 修改架构

### 核心流程

```
输入图像 → CLIP ViT → 各层特征
                    │
                    ├──→ 原始分类损失 (CrossEntropy)
                    │
                    └──→ SVD 正交约束
                         │
                         ├──→ 对冻结的 Attention 权重做 SVD
                         ├──→ 提取前 K 个主成分 (语义基)
                         └──→ 计算 WSGM 输出在语义子空间的投影

总损失 = loss_cls + λ × loss_orth
```

---

## 📝 文件修改清单

### 1. `models/network/clip/model.py` - 核心实现

#### 新增参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `svd_rank` | 64 | 语义子空间秩（主成分数量）|
| `svd_energy_ratio` | 0.9 | SVD 能量保留比例 |

#### 主要修改

**ResidualAttentionBlock** (第 264-289 行)
- 新增 `semantic_proj_matrix` 属性
- 新增 `return_orth_loss` 参数
- 计算 WSGM 输出在语义子空间的投影作为正交损失

```python
def forward(self, x: torch.Tensor, return_orth_loss: bool = False):
    # ... 标准前向 ...
    if self.use_wsgm and self.wsgm_module is not None:
        wsgm_output = self.wsgm_module(x)
        x = x + wsgm_output

        # 计算正交损失
        if return_orth_loss and self.semantic_proj_matrix is not None:
            wsgm_flat = wsgm_output.permute(1, 0, 2).reshape(-1, d_model)
            proj = torch.matmul(wsgm_flat, self.semantic_proj_matrix)
            orth_loss = torch.mean(proj ** 2).detach()
```

**Transformer** (第 330-387 行)
- 新增 `_compute_semantic_projection_matrix()` 方法
- 对冻结的 Attention 权重进行 SVD 分解
- 计算累积能量，确定语义子空间秩

```python
def _compute_semantic_projection_matrix(self) -> torch.Tensor:
    """计算语义投影矩阵"""
    # 收集所有层的 in_proj_weight
    all_weights = [block.attn.in_proj_weight for block in self.resblocks]
    combined_weights = torch.cat(all_weights, dim=0)

    # SVD 分解
    U, S, Vh = torch.linalg.svd(combined_weights, full_matrices=False)
    V = Vh.T

    # 按能量比例确定秩
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    rank_by_energy = torch.searchsorted(cumulative_energy, self.svd_energy_ratio) + 1
    rank = min(self.svd_rank, rank_by_energy, V.size(1))

    return V[:, :rank].float()  # [d_model, rank]
```

**VisionTransformer & CLIP**
- 新增 `return_orth_loss` 参数传递
- 支持可选返回正交损失

---

### 2. `models/network/net_stage1.py` - 模型接口

**修改位置**: 第 33-46 行

```python
def forward(self, x, return_rcs=False, return_orth_loss: bool = False):
    if return_rcs:
        feature, cls_tokens, rcs_token = self.backbone.encode_image(x, return_rcs=True)
        result = self.fc(feature)
        return result, cls_tokens, rcs_token
    else:
        if return_orth_loss:
            feature, cls_tokens, orth_loss = self.backbone.encode_image(x, return_orth_loss=True)
            result = self.fc(feature)
            return result, cls_tokens, orth_loss
        else:
            feature, cls_tokens = self.backbone.encode_image(x, return_rcs=False)
            result = self.fc(feature)
            return result, cls_tokens
```

---

### 3. `models/trainer_stage1.py` - 训练集成

**新增配置** (第 31-33 行)
```python
self.orth_lambda = getattr(opt, 'orth_lambda', 0.1)  # 正交损失权重
print(f"[SVD Orthogonal Loss] Lambda coefficient: {self.orth_lambda}")
```

**训练循环** (第 47-59 行)
```python
with autocast():
    output, _, orth_loss = self.model(data, return_orth_loss=True)

    # 分类损失
    loss_cls = criterion(output.squeeze(1), target.type(torch.float32))

    # 总损失
    if orth_loss is not None:
        loss = loss_cls + self.orth_lambda * orth_loss
    else:
        loss = loss_cls
```

**验证循环** (第 76-116 行)
- 额外记录 `running_orth_loss`
- TensorBoard 新增 `Loss_Orth/Validation` 日志

---

### 4. `options/options.py` - 命令行参数

**新增参数** (第 28-34 行)
```python
# SVD Orthogonal Constraint (Effort paper)
parser.add_argument('--orth_lambda', type=float, default=0.1,
                    help='Weight for SVD orthogonal loss in Stage 1')
parser.add_argument('--svd_rank', type=int, default=64,
                    help='Rank of semantic subspace (number of principal components)')
parser.add_argument('--svd_energy_ratio', type=float, default=0.9,
                    help='Energy ratio for SVD rank selection (0-1)')
```

---

### 5. `prepare_small_dataset.py` - 数据集准备

**修改内容**: 改为按比例采样

| 修改前 | 修改后 |
|--------|--------|
| `samples_per_class=100` | `percentage=0.01` |
| 固定数量 | 按百分比自动计算 |

```python
# 1% 数据
setup_small_dataset(percentage=0.01)

# 10% 数据
setup_small_dataset(percentage=0.1)

# 20% 数据
setup_small_dataset(percentage=0.2)
```

---

## 🎯 关键设计决策

### 1. 保护 RCS Token

- `manual_attn()` 方法完全保留
- Stage 2 评估时不需要正交损失
- 通过 `return_orth_loss` 参数控制是否计算正交损失

### 2. 效率优化

- **SVD 只计算一次**: 在 `Transformer.__init__` 中完成
- **梯度分离**: 使用 `.detach()` 确保正交损失不影响主计算图
- **设备处理**: 动态将 `semantic_proj_matrix` 移到正确设备

### 3. 数学原理

根据 Effort 论文：
- 冻结的 CLIP Attention 权重主成分方向 → 语义子空间
- WSGM 输出在这些方向上的投影 → 应该最小化
- 最小化投影 = 强制 WSGM 学习与语义正交的特征 = 挖掘伪造细节

---

## 📊 实验参数

### 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `orth_lambda` | 0.1 | 正交损失权重 |
| `svd_rank` | 64 | 语义子空间秩 |
| `svd_energy_ratio` | 0.9 | 保留 90% 能量 |

### 数据集比例 (论文设置)

| 比例 | 样本数 (4类) | 用途 |
|------|-------------|------|
| 1% | ~1,440 | 快速验证 |
| 10% | ~14,400 | 消融实验 |
| 20% | ~28,800 | 中等规模 |
| 50% | ~72,000 | 完整训练 |
| 100% | ~144,000 | 全量数据 |

---

## ✅ 兼容性

| 功能 | 状态 |
|------|------|
| RCS Token 提取 | ✅ 正常 |
| Stage 1 训练 | ✅ 正常 |
| Stage 2 训练 | ✅ 正常 |
| 模型评估 | ✅ 正常 |
| 断点续训 | ✅ 兼容 |

---

## 🚀 使用方法

### 快速开始 (1% 数据)

```bash
# 1. 准备数据集
python prepare_small_dataset.py

# 2. 训练 (含 SVD 正交约束)
bash train_setting_1.sh
```

### 自定义参数

```bash
# 调整正交损失权重
python train.py --orth_lambda 0.05 --orth_lambda 0.2

# 调整 SVD 秩
python train.py --svd_rank 32 --svd_energy_ratio 0.95
```

---

## 📈 预期效果

1. **防止特征冗余**: WSGM 不会重复学习 CLIP 已有的语义特征
2. **专注伪造痕迹**: 强制在语义子空间的补空间（伪造子空间）学习
3. **提升泛化能力**: 更好捕捉跨域伪造特征

---

## 🔧 后续优化方向

1. **自适应 λ**: 根据训练进度动态调整 `orth_lambda`
2. **层级别权重**: 不同层使用不同的正交约束强度
3. **多尺度正交**: 在不同特征粒度上应用正交约束

---

**文档版本**: 1.0
**最后更新**: 2026年1月
