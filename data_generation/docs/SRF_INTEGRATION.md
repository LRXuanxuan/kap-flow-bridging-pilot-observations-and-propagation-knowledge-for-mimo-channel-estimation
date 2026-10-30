# CKM-conditioned SRF 接入建议

## 1. 推荐模型输入

对每个快照构建：

```text
channel state: H_t
flow time: t
continuous position: observedPositionXYZ
velocity/orientation: velocityXYZ, ueOrientationDeg
CKM scalar feature: scalarFeature
CKM angular-delay feature: angularDelayPower
CKM beam feature: txBeamProbability, rxBeamProbability
```

位置建议先归一化到 `[-1,1]`，再使用 Fourier positional encoding。
CKM feature 可通过 MLP/1-D encoder 编码，并使用 FiLM 或 adaLN 注入流网络。

## 2. 条件 DPR

将原模型：

```text
mu_theta(H_t, t)
```

改为：

```text
mu_theta(H_t, t, c_ckm)
```

对应干净信道中心：

```text
H0_hat = H_t - t * mu_theta(H_t, t, c_ckm)
```

CKM 条件主要影响导频零空间。闭式导频一致性和 PBG 仍由 `Y,P`
驱动，不建议将测试瞬时路径真值输入 PBG。

## 3. 原始 SRF 模式

当 `cfo.enable=false`：

```text
A_P(H) = H P
A_P^*(R) = R P^H
```

当前 SRF 的闭式 proximal update 和 PBG 可以直接使用。

## 4. CFO/ICI 广义模式

启用 CFO 后：

```text
Y_k = exp(j phi_t) sum_m C_km(epsilon) H_m P + N_k
```

PBG 应使用：

```text
X_next = X - eta * A^*(A(X)-Y_bridge)
```

MATLAB 参考实现：

```matlab
Y = ckmsrf_apply_operator(H,P,cfoHz,cfg);
gradient = ckmsrf_apply_adjoint(residual,P,cfoHz,cfg);
```

Python/PyTorch 版本应通过相同的复数伴随测试。

## 5. 推荐训练损失

除原 DPR loss 外，可随机采样导频矩阵并增加：

```text
L_null = ||Pi_N(H0_hat-H0)||_F^2 / (||Pi_N(H0)||_F^2+eps)
```

可选辅助损失：

```text
L_beam     top-K beam distribution loss
L_pathloss CKM large-scale gain regression loss
```

建议主指标同时报告：

```text
NMSE_total, NMSE_range, NMSE_null, pilot residual, achievable rate
```
