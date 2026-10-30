# CKM-SRF 统一数据生成器

本目录是 KAP-Flow 论文所用三站点移动毫米波 MIMO 数据集的 MATLAB 生成器。
工程仅依赖基础 MATLAB（**不依赖 5G Toolbox**），可在本地合成：

- 统一宽带 MIMO 信道张量 `H(rx, tx, subcarrier, snapshot, sample)`；
- 仅由**独立历史信道样本**构建的位置索引 CKM，以及沿轨迹的双线性 CKM 查询；
- 有限导频观测 `Y = H P + N`（半酉 DFT 导频矩阵）；
- 路径级真值、UE 状态，以及空间不重叠的 train/validation/test 划分。

KAP-Flow 直接读取导出的 **plain HDF5** 分片（`data.backend: plain_h5`），
不需要把 `.mat` 转换成 SRF 的训练格式；原始 MATLAB `.mat` 分片也可通过可选的
`mat_shard_stream` 后端读取。数据契约与论文数据设置详见
[`../docs/DATA.md`](../docs/DATA.md)。

> 数据集本体（数百 GiB）**不随代码分发**：请用本生成器在本地生成，或把
> `CKMSRF_DATA_ROOT` 指向符合契约的 HDF5 数据根目录。

## 核心能力

- 统一信道维度 `H(rx, tx, subcarrier, snapshot, sample)`；论文配置为
  `(N_t, N_r, N_f, N_s) = (64, 16, 48, 8)`；
- 几何定义的多站点场景：固定散射体/阻挡物、LoS/NLoS、半波长 `8x8` 发射与
  `4x4` 接收 UPA、`f_c = 28 GHz`、`Delta f = 15 kHz`；
- 高移动序列：恒速 UE 轨迹（`20--60 m/s`），同一轨迹内物理 `pathId` 固定，
  时延随时间变化、含载波多普勒；
- 由独立历史样本构建 CKM：每个节点聚合 32 条历史信道为 `16x16x12`
  角度-时延功率谱，另含 6 维标量特征与 64/16 收发波束概率；在线信道不参与建图；
- 空间块级划分：`40 m x 40 m` 块按 `15/5/5` 分给 train/val/test，
  同一条轨迹的所有快照始终属于同一划分；
- 正确的有限导频观测 `Y = H P + N`，`P^H P = I_{N_p}`（论文配置 `N_p = 19`）；
  每条记录保存导频编号、SNR 与噪声方差；
- shard 分片保存以控制内存，另写 `manifest.mat`；提供显式 HDF5 导出；
- 可选 CFO/ICI 广义线性算子及其精确伴随（论文数据关闭：
  `cfg.cfo.enable = false`）。

字段级 schema 见 [`docs/SCHEMA.md`](docs/SCHEMA.md)；SRF 时代的接入笔记
[`docs/SRF_INTEGRATION.md`](docs/SRF_INTEGRATION.md) 仅作历史参考，
KAP-Flow 的接入方式以 [`../docs/DATA.md`](../docs/DATA.md) 为准。

## 目录结构

```
data_generation/
├── run_demo.m                      # 小规模端到端演示
├── generate_ckm_srf_dataset.m      # 顶层生成器（输出 .mat 分片）
├── ckmsrf_default_config.m         # 冒烟测试默认配置
├── configs/research_config.m       # 论文规模配置
├── export_scene_to_hdf5.m          # 场景 .mat  -> plain HDF5
├── export_shard_to_hdf5.m          # 分片 .mat  -> plain HDF5
├── export_all_to_hdf5.m            # 对整个输出目录批量导出
├── src/*.m                         # 几何、CKM、轨迹、导频、I/O 辅助函数
├── tests/run_all_tests.m           # MATLAB 物理与 schema 测试
├── python/                         # HDF5 读取器、示例、KAP-Flow 数据准备脚本
└── docs/                           # 生成器字段说明与历史接入笔记
```

## 环境要求

- MATLAB，需支持 `save(..., '-v7.3')`、`h5create`、`h5write`；不需要任何工具箱。
- Python >= 3.10（HDF5 读取与 KAP-Flow 数据 loader）：
  `python -m pip install -r python/requirements.txt`。

## 快速开始（MATLAB 演示）

```matlab
cd('<repo>/data_generation')
run_demo                                % 1 个场景，12/6/6 条 train/val/test 记录
run('tests/run_all_tests.m')            % 物理与 schema 测试
```

演示结果写到 `data_generation/` 下的 `output_demo/`，已排除在 git 之外。
导出为 HDF5 并冒烟测试 Python 读取器（shell 命令在仓库根目录执行）：

```matlab
export_all_to_hdf5('<repo>/data_generation/output_demo');   % 输出 output_demo_h5/
```

```bash
python -m pip install -r data_generation/python/requirements.txt
python data_generation/python/test_h5_loader.py \
    data_generation/output_demo_h5/scene_01_train_shard_001.h5 \
    data_generation/output_demo_h5/scene_01_ckm.h5
```

## 生成论文规模数据

`configs/research_config.m` 对应论文第 IV 节的数据设置（三站点、`200 m x 200 m`、
10 m CKM 网格、`15/5/5` 空间块划分、每站点 `10000/1000/2000` 条记录、
16 个 DFT 导频矩阵、SNR `-10:5:30` dB）；完整数字见
[`../docs/DATA.md`](../docs/DATA.md)。

```matlab
addpath('<repo>/data_generation');
addpath('<repo>/data_generation/configs');
addpath('<repo>/data_generation/src');

% 1) 先用极小样本数在独立目录验证磁盘/内存/版本兼容
smoke = research_config('<datasets>/ckm_srf_smoke');
smoke.dataset.samplesPerScene.train = 10;
smoke.dataset.samplesPerScene.val   = 2;
smoke.dataset.samplesPerScene.test  = 2;
smoke.overwrite = true;
manifest = generate_ckm_srf_dataset(smoke);

% 2) 再跑完整论文规模（需要较长时间与较大磁盘）
cfg = research_config('<datasets>/ckm_srf_64x16');
manifest = generate_ckm_srf_dataset(cfg);
```

必须保持 `cfg.output.storeTrueChannel = true`。`cfg.output.storeObservation`
建议保持 `false`：KAP-Flow 会用干净信道与导频库在线合成 `Y`，而每条记录的
导频编号/SNR/噪声方差仍会保存。存储量随干净信道张量增长：在 48 子载波、
8 快照、16x64 天线的 complex single 下，一条记录恰为 `3 MiB`；一个站点
10,000 条训练记录约 `29.3 GiB`（尚未计 HDF5 开销）。

批量导出为 plain HDF5：

```matlab
export_all_to_hdf5('<datasets>/ckm_srf_64x16');   % -> <datasets>/ckm_srf_64x16_h5
```

## 把数据喂给 KAP-Flow

KAP-Flow 通过环境变量 `CKMSRF_DATA_ROOT`（或 `--data-root`）定位数据根，
要求 HDF5 文件位于该目录顶层：

```
<data root>/
├── scene_01_ckm.h5                 # 导频库 + 离线 CKM（每场景一个）
├── scene_01_train_shard_001.h5
├── scene_01_val_shard_001.h5
├── scene_01_test_shard_001.h5
├── scene_02_ckm.h5  scene_02_*_shard_*.h5
├── scene_03_ckm.h5  scene_03_*_shard_*.h5
├── indices/{train,val,test}.json   # 可选：准备脚本生成
└── normalization.json              # 可选：仅训练集统计量
```

训练/评估 loader 能在缺少时自动构建 `indices/` 与统计量。为可复现，建议显式运行：

```bash
# 在仓库根目录；先安装 pip install -e .
export CKMSRF_DATA_ROOT="<datasets>/ckm_srf_64x16_h5"
python data_generation/python/prepare_kapflow_data.py --data-root "$CKMSRF_DATA_ROOT"
```

该脚本会校验 HDF5 字段、修正 MATLAB 根属性（见下文“兼容性说明”）、
生成 `indices/{split}.json`，并用训练集计算 `normalization.json`，
最后打印下一步训练命令。

```bash
python -m ckmsrf.train \
    --config configs/experiments/ckm_straight_flow_prior.yaml \
    --data-root "$CKMSRF_DATA_ROOT"
```

跨场景（留一站）统计量需用对应源场景计算，并写入配对的统计量文件名，
例如 `holdout_01`（用场景 02/03 训练）：

```bash
python data_generation/python/prepare_kapflow_data.py \
    --data-root "$CKMSRF_DATA_ROOT" --scenes 2 3 \
    --normalization-file normalization_cross_scene_01.json
```

论文 SNR 网格上的评估示例与完整协议见顶层 README；数据与权重都不入 git。

## HDF5 与 MATLAB `.mat` 分片

| 项目 | Plain HDF5（`plain_h5`） | 原始 `.mat`（`mat_shard_stream`） |
|---|---|---|
| 由谁产出 | `export_*_to_hdf5.m` | `generate_ckm_srf_dataset.m` |
| Python 依赖 | `h5py` | `hdf5storage`（可选） |
| 访问方式 | 逐样本惰性切片 | 整 shard 读入内存 |
| 推荐用途 | 训练与评估 | 原始输出快速检查 |

MATLAB 高层 HDF5 接口写出的数组在 h5py 中呈逆序轴；KAP-Flow loader 与
`python/ckmsrf_h5_dataset.py` 会恢复逻辑顺序
`(N_r, N_t, N_f, N_s, N)`。原始 shape 对照见
[`../docs/DATA.md`](../docs/DATA.md)。

## 兼容性说明（导出前必读）

`export_*_to_hdf5.m` 保持生成器原样，其中两处根属性约定与 KAP-Flow loader
不一致：

1. `h5writeatt(..., 'scene_id', uint16(...))` 写出的是**单元素数组**，
   而 loader 用 `int(...)` 读取；未修正时 `build_shard_index` 会抛
   `TypeError`；
2. 导出器未写 `storage_order` 属性。loader 的形状启发式对导出布局正确，
   但当记录数恰等于 `N_r`（16）时存在歧义。

`python/prepare_kapflow_data.py` 会自动修复这两点（幂等；可用
`--check-only` 只检查不修改）。若自行构建 HDF5，请保持导出器约定并在数据根
上运行一次准备脚本。

## 数据边界

- `truth/*` 只是监督标签/分析变量；
- 在线模型条件只能来自 `state/*` 与 `context/*`；
- `context/*` 由**独立历史** CKM 插值得到，不含当前样本的瞬时路径增益；
- `aux/*`（BIM/波束标签）是辅助标签，不得作为无误差 oracle 输入；
- 同一轨迹的全部快照属于同一空间划分。

## See also

- 数据契约与论文数据设置：[`../docs/DATA.md`](../docs/DATA.md)
- 生成器字段说明：[`docs/SCHEMA.md`](docs/SCHEMA.md)
- 仓库总览（安装、训练、评估）：顶层 [`README.md`](../README.md)
