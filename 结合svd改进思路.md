
# ForgeLens SVD-Orthogonal 增强实施指南 (Stage 1)

## 📄 文档概述

本文档旨在指导如何在 **ForgeLens Stage 1 (WSGM 训练阶段)** 中引入基于 **Effort (Orthogonal Subspace Decomposition)** 的 SVD 正交约束机制。
**核心目标**：通过数学约束，强制 WSGM 学习与 CLIP 预训练语义特征（Semantic Subspace）**正交**的残差特征（Forgery Subspace），从而避免语义重复学习，提升对伪造痕迹的捕获能力。

> ⚠️ **重要警告**：本项目已包含 Stage 2 的 RCS (Residual Context Structure) 增强。在修改底层代码时，**必须确保 `manual_attn` 等服务于 RCS 的接口不被破坏**。

---

## 🛠 修改清单与实施细节

### 1. 修改 `models/network/clip/model.py`

这是核心修改点。我们需要在 `ResidualAttentionBlock` 中“埋入”SVD 投影矩阵，并在前向传播时计算正交损失。

#### 1.1 新增：SVD 初始化方法 (`init_svd_projection`)

在 `ResidualAttentionBlock` 类中添加此方法。它只应在 Stage 1 开始前被调用一次。

```python
    def init_svd_projection(self, energy_threshold=0.90):
        """
        [SVD-Orthogonal Action]
        对 Self-Attention 的输入投影权重进行 SVD 分解，构建语义投影矩阵。
        该矩阵作为 Buffer 注册，不参与梯度更新。
        """
        # 1. 获取 Attention 的投影权重 (Q, K, V combined)
        # 注意: CLIP 的 MultiheadAttention 通常使用 in_proj_weight
        weight = self.attn.in_proj_weight.detach() 
        
        # 2. 执行 SVD 分解
        # U: (D_out, D_out), S: (min_D,), V: (D_in, D_in)
        # 这是一个计算密集型操作，仅需运行一次
        try:
            U, S, V = torch.linalg.svd(weight, full_matrices=False)
        except:
            # 兼容旧版 PyTorch
            U, S, V = torch.svd(weight)

        # 3. 确定保留的主成分数量 (Top-K)
        # 基于能量占比 (Energy Ratio)
        energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
        k = torch.searchsorted(energy, energy_threshold).item() + 1
        # 或者设置一个硬阈值，例如 k = min(k, 160)
        
        # 4. 构建语义投影矩阵 P_sem = U_k * U_k^T
        # Shape: (D_model, D_model)
        U_k = U[:, :k]
        P_sem = torch.mm(U_k, U_k.t())
        
        # 5. 注册为 Buffer (Persistent=False 表示可能不需要存入 state_dict，视需求而定)
        self.register_buffer('sem_proj', P_sem)
        print(f"Initialized SVD Projection: Kept {k} components ({energy_threshold*100}%)")

```

#### 1.2 修改：Forward 函数 (`forward`)

我们需要在不干扰 RCS 逻辑 (`manual_attn`) 的前提下，计算 WSGM 输出的正交性。

```python
    # 修改 forward 函数签名，增加 return_orth_loss 参数
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_orth_loss: bool = False):
        # --- 原有逻辑 (RCS 依赖此部分，勿动) ---
        # self.attn(...) 等标准流程
        # ...
        
        # --- WSGM 逻辑 (ForgeLens 插入点) ---
        # 假设 WSGM 已经注入到 Block 中，通常名为 self.wsgm 或类似
        # 原始: x = x + self.wsgm(x_ln) 
        
        # 获取 WSGM 的纯特征输出 (不加残差前)
        wsgm_output = self.wsgm(self.ln_1(x)) # 或者是 wsgm 内部计算出的特征
        
        # --- 新增: 正交损失计算 ---
        orth_loss = None
        if return_orth_loss and hasattr(self, 'sem_proj'):
            # 计算 WSGM 特征在语义空间上的投影
            # wsgm_output: [Batch, Seq, Dim]
            # sem_proj:    [Dim, Dim]
            # project:     [Batch, Seq, Dim]
            projection = torch.matmul(wsgm_output, self.sem_proj)
            
            # 我们希望投影量越小越好 (即正交)
            # 使用 L2 范数的平方作为 Loss
            orth_loss = torch.norm(projection, p=2) ** 2 / projection.numel()
        
        # 应用 WSGM 残差
        x = x + wsgm_output
        
        # MLP 部分 (保持不变)
        x = x + self.mlp(self.ln_2(x))

        if return_orth_loss:
            return x, orth_loss
        return x

```

---

### 2. 修改 `models/network/net_stage1.py` (桥接层)

`NetStage1` 通常作为 CLIP 的 Wrapper。我们需要在这里暴露 SVD 初始化接口，并在 Forward 中聚合所有层的 Loss。

```python
class NetStage1(nn.Module):
    # ... 现有代码 ...

    def init_svd_for_training(self):
        """遍历所有 Block 并初始化 SVD"""
        print("Initializing SVD constraints for CLIP Blocks...")
        for block in self.image_encoder.transformer.resblocks:
            if hasattr(block, 'init_svd_projection'):
                block.init_svd_projection()
                
    def forward(self, x, return_loss=False):
        # 在调用 image_encoder 时，传递 return_orth_loss=True
        # 这可能需要修改 CLIP 的 forward 或者手动遍历 blocks
        
        # 建议方案：如果 CLIP 代码难以改动 forward 签名，
        # 可以用 hook 或者在 NetStage1 里手动循环 resblocks
        
        features = x
        total_orth_loss = 0.0
        
        # 手动执行 Transformer 层以捕获 loss (伪代码)
        for i, block in enumerate(self.image_encoder.transformer.resblocks):
            if self.training:
                features, loss = block(features, return_orth_loss=True)
                if loss is not None:
                    total_orth_loss += loss
            else:
                features = block(features)
        
        # ... 后续分类头逻辑 ...
        
        if self.training and return_loss:
            return logits, total_orth_loss
        return logits

```

---

### 3. 修改 `models/trainer_stage1.py` (训练循环)

最后，将 Loss 整合到优化步骤中。

```python
    # 在 Trainer 初始化或训练开始前调用
    def before_train_loop(self):
        # 确保只计算一次 SVD
        self.model.init_svd_for_training()

    def train_step(self, batch):
        # ... 数据加载 ...
        
        # Forward
        logits, orth_loss_sum = self.model(images, return_loss=True)
        
        # 计算原本的分类 Loss
        loss_cls = self.criterion(logits, labels)
        
        # 计算总 Loss
        # 建议 lambda 系数: 0.1 ~ 0.01
        lambda_orth = 0.05 
        
        # 如果 orth_loss_sum 是所有层的和，可能需要除以层数做归一化
        num_layers = len(self.model.image_encoder.transformer.resblocks)
        avg_orth_loss = orth_loss_sum / num_layers
        
        total_loss = loss_cls + lambda_orth * avg_orth_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Logging
        # 记录 loss_orth 以便观察收敛情况

```

---

## 🔍 验证与检查点 (Checklist)

在代码修改完成后，请进行以下检查以确保“无侵入”原则得到遵守：

1. **Stage 2 兼容性检查**：
* 运行 `evaluate.sh` (基于 Stage 2 RCS)。
* 确保代码不会报错（因为 Stage 2 运行时 `return_orth_loss` 默认为 `False`，且 SVD buffer 即使存在也不会被使用）。
* **预期**：Stage 2 的推理结果应与修改前完全一致（0 error deviation）。


2. **SVD 缓存检查**：
* 在训练启动日志中，检查是否输出了 `"Initialized SVD Projection..."`。
* 确保该日志只在程序启动时出现一次，而不是每个 Batch 都出现。


3. **Loss 观察**：
* 使用 TensorBoard 或日志观察 `loss_orth`。
* 它应该随着训练逐渐下降，这表明 WSGM 正在学着“避开”CLIP 的主成分方向，向残差空间迁移。