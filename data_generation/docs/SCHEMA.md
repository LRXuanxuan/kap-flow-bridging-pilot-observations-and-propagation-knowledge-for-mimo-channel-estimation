# 数据结构与维度

## 场景文件

`scene_XX_ckm.mat` 包含变量 `scenePackage`：

| 字段 | 含义 |
|---|---|
| `scene` | BS、固定散射体、反射系数、阻挡线段 |
| `ckm` | 由历史样本构建的空间先验 |
| `splitPlan` | 空间 block 到 train/val/test 的映射 |
| `pilotBank` | 半酉 DFT 导频矩阵 |
| `Ftx`, `Wrx` | 发射和接收 DFT 波束码本 |

CKM scalar feature 的固定顺序为：

```text
1  LoS probability
2  pathloss mean (dB)
3  pathloss standard deviation (dB)
4  mean delay (s)
5  RMS delay spread (s)
6  number of historical samples
```

`angularDelayPower` 展平前的顺序由 `ckm.adpShape` 给出：

```text
[numAodBins, numAoaBins, numDelayBins]
```

## shard 文件

每个 shard 保存变量 `shard`。

### 主信道和观测

| 字段 | 维度 |
|---|---:|
| `H_true` | `Nr x Nt x Nsc x T x N` |
| `Y_pilot` | `Nr x Np x Nsc x T x N` |
| `channelScaleLin` | `Nsc x T x N` |

如配置关闭存储，相应字段为空数组。

### 状态与 CKM 条件

| 字段 | 维度 |
|---|---:|
| `state.positionXYZ` | `3 x T x N` |
| `state.velocityXYZ` | `3 x T x N` |
| `state.ueOrientationDeg` | `1 x T x N` |
| `state.blockId` | `T x N` |
| `context.observedPositionXYZ` | `3 x T x N` |
| `context.gridIndices` | `4 x T x N` |
| `context.gridWeights` | `4 x T x N` |
| `context.scalarFeature` | `6 x T x N` |
| `context.angularDelayPower` | `F_adp x T x N` |
| `context.txBeamProbability` | `Nt x T x N` |
| `context.rxBeamProbability` | `Nr x T x N` |

`state.positionXYZ` 是真值位置；模型可使用的是
`context.observedPositionXYZ`。当 `localization.errorStdM>0` 时，二者不同。

### 路径真值

所有路径字段以 `L x T x N` 保存：

```text
pathId, pathMask, pathGain, pathPowerLin, delaySec,
aodAzRad, aodElRad, aoaAzRad, aoaElRad, dopplerHz
```

同一轨迹中 `pathId(:,t,n)` 固定，消失路径通过 `pathMask=false` 表示。

### 导频信息

| 字段 | 维度 |
|---|---:|
| `observation.pilotId` | `1 x N` |
| `observation.snrDb` | `1 x N` |
| `observation.cfoHz` | `1 x N` |
| `observation.noiseVar` | `1 x N` |
| `sampleSeed` | `1 x N` |

真实导频矩阵通过场景文件中的：

```matlab
P = scenePackage.pilotBank.P(:,:,pilotId);
```

获取。

## HDF5 与 Python 轴顺序

MATLAB 高层 HDF5 接口写出的 dataset 在 `h5py` 中呈现为逆序轴。例如，
`H_true: Nr x Nt x Nsc x T x N` 会被原始 `h5py` 读取为
`N x T x Nsc x Nt x Nr`。必须通过 `python/ckmsrf_h5_dataset.py` 读取，
或对所有轴做逆序转置。工程提供的加载器输出始终遵循本页的 MATLAB/schema
维度约定。
