# Method: knowledge-and-state conditioning (KSC) and pilot--prior consensus (PPC)

Reference description of the method implemented in `src/ckmsrf/` for the paper

> **KAP-Flow: Bridging Pilot Observations and Propagation Knowledge for MIMO Channel Estimation**
> (manuscript under review at the time of this release; see `CITATION.cff`)

Equation numbers refer to the submitted manuscript; section numbers refer to
its sections (II-A ... IV-D). Code names are not always the paper names: the
paper method **KAP-Flow (KAP)** is the code method **`flowadmm`** (legacy name
*FlowADMM*), and **PnP-Flow (PF)** is `pnpflow`. The complete mapping is in
[`docs/NAMING.md`](NAMING.md). Everything marked *see code* is a pointer, not a
claim about an undocumented implementation detail.

## 0. Notation and conventions

| Paper | Code | Meaning |
| --- | --- | --- |
| $\mat H\in\mathbb C^{N_r\times N_t}$ | `batch["H"]`, `truth` | downlink channel (complex `[B,Nr,Nt]`) |
| $\mat P\in\mathbb C^{N_t\times N_p}$ | `batch["P"]`, `pilot` | pilot matrix, $N_p<N_t$ |
| $\mat Y$, $\nu$ | `batch["Y"]`, `batch["noise_var"]` | observation and complex noise variance |
| $\mat h=T(\mat H)$ | two-channel tensor `[B,2,Nr,Nt]` | real--imaginary (isometric) representation |
| $\vct\epsilon$ | `torch.randn` latent | flow latent noise (unit variance) |
| $s$ (artificial flow time) | `t` / `time_batch` | paper uses $s\in[0,1]$, code uses `t` |
| $M$ | `latent_samples` / `k_avg` | endpoints averaged per PPC iteration |
| $K$ | `iterations` / `--pnp-iterations` | ADMM outer iterations |
| $\tau$ | `tau` / `rho = 1/tau` | inverse ADMM penalty |
| $D_{s,\mat c}(\mat x)$ | `denoise(...)` | one-step conditional endpoint predictor |

Code convention: `torch.view_as_real` two-channel tensors are ordered
`[real, imag]`; the complex operator conventions are
`A(H)=H P` and `A^H(R)=R P^H` (`src/ckmsrf/physics/observation.py`).

## 1. Pilot observation model (paper Sec. II-A, Eq. (2))

Paper:
```
Y_{k,n} = H_{k,n} P + N_{k,n},    [N_{k,n}]_ab ~ CN(0, nu_{k,n})
```

| Paper element | Code |
| --- | --- |
| $A(\mat H)=\mat H\mat P$ | `src/ckmsrf/physics/observation.py::apply_measurement` |
| $A^{\mathrm H}(\mat R)=\mat R\mat P^{\mathrm H}$ | `src/ckmsrf/physics/observation.py::apply_adjoint` |
| $\mat Y=\mat H\mat P+\mat N$ with variance from $\xi$ | `src/ckmsrf/physics/observation.py::generate_observation` |
| per-sample noise variance used by the solvers | `cache["sigma2"]` in `src/ckmsrf/physics/rc.py::build_physics_cache` |
| complex $\leftrightarrow$ two-channel conversion | `src/ckmsrf/physics/observation.py::complex_to_twochannel`, `::twochannel_to_complex` |
| offline dataset-side observation synthesis (shard generation) | `src/ckmsrf/data/observation.py::apply_pilot`, `::synthesize_observation`, `::add_complex_awgn` |
| noise-normalized pilot residual (training auxiliary) | `src/ckmsrf/physics/observation.py::data_consistency_loss` |

Notes.

* The paper's evaluation variance
  $\nu=\lVert\mat H\mat P\rVert_F^2/(N_rN_p10^{\xi/10})$ is exactly
  `signal_power = clean.abs()**2.mean((-2,-1))`, `variance = signal_power / 10**(snr_db/10)`
  in `generate_observation`, where `clean = H @ P`.
* The manuscript states that the experiments use $\mat P^{\mathrm H}\mat P=\mat I$.
  The code does **not** assume semi-unitary pilots: it stores
  `p_pinv = pinv(P)` and solves the general normal equations
  (see Section 6).

## 2. CKM and location query (paper Sec. II-B; descriptor vector in Sec. III-B, Eq. (7))

A location-indexed channel knowledge map stores, per grid node $g$, the
descriptor
```
m_g = [s_g; vec(A_g); b_g^tx; b_g^rx],
```
with six scalar statistics $s_g$ (LoS occurrence, log-power mean/std, mean
delay, RMS delay spread, sample count), a normalized $16\times16\times12$
angular--delay power histogram $A_g$, and 64/16 DFT-beam preference vectors.
For a UE at $\widehat{\mat q}(t_n)$ the local descriptor $\mat m_n$ is the
bilinear interpolation of the four surrounding nodes.

| Paper element | Code / data contract |
| --- | --- |
| descriptor $\mat m_g$ | HDF5 context fields named in `src/ckmsrf/data/lazy_dataset.py::_plain_sample`: `/context/scalar_feature` (6), `/context/angular_delay_power` ($16{\times}16{\times}12$), `/context/tx_beam_probability` (64), `/context/rx_beam_probability` (16) |
| interpolated descriptors $\mat m_n$ | read per sample as the same `/context/*` fields; the dataset also carries the interpolation metadata `/context/grid_indices` (4 node ids) and `/context/grid_weights` (4 weights) |
| bilinear interpolation itself | performed offline by the CKM data generator; see `docs/DATA.md` and `data_generation/` |
| site/UE state per record | `/state/position_xyz`, `/state/velocity_xyz`, `/state/ue_orientation_deg`, `/context/observed_position_xyz` |
| feature normalization | `src/ckmsrf/data/normalization.py` (`global_rms`; per-sample `noise_var` is divided by `h_rms**2`) |

The interpolation weights are not recomputed at training or inference time;
they are part of the frozen map (`experiment` note: "CKMs remain fixed at
inference", manuscript Sec. IV-A).

## 3. Flow-matching preliminaries (paper Sec. II-C)

Paper: straight path between latent noise and clean CSI,
```
h = T(H),    x_s = (1-s) eps + s h,     eps ~ N(0, I),
L_FM(theta) = E_{pi,s} || v_theta(x_s, s) - (h - eps) ||^2,
```
and the one-step endpoint predictor `D_s(x) = x + (1-s) v_theta(x, s)`.

| Paper element | Code |
| --- | --- |
| straight path $x_s=(1-s)\vct\epsilon+s\mat h$ | `src/ckmsrf/losses/straight_flow.py::StraightFlowMatchingObjective.forward` (`path_state = (1-time)*x_latent + time*x_data`) |
| target velocity $\mat h-\vct\epsilon$ | `::StraightFlowMatchingObjective.forward` (`target_velocity = x_data - x_latent`) |
| coupling $\pi$ (mini-batch OT) | `::StraightFlowMatchingObjective._minibatch_ot_permutation` (exact uniform assignment, Hungarian; only the latent samples are permuted so $(\mat h,\mat c)$ pairs stay aligned; internal note in code) |
| endpoint predictor $D_s$ | `src/ckmsrf/models/straight_flow.py::ConditionalStraightFlowUNet.denoise` (`x + (1-t)*velocity`) |
| same predictor inside the solvers | `src/ckmsrf/sampling_flow.py::_average_renoise_denoise` |
| velocity network | `src/ckmsrf/models/straight_flow.py::ConditionalStraightFlowUNet` (`output_parameterization = "straight_velocity"`) |
| objective/optimizer wiring | `src/ckmsrf/factory.py::build_model`, `::build_objective`; `src/ckmsrf/train.py` |

Important: this repository also contains the older **additive/VE** flow
objective (`src/ckmsrf/losses/gmflow.py::GMFlowObjective`, used by the
RC/SRF baselines). The straight-flow solvers explicitly reject non-straight
checkpoints (`src/ckmsrf/sampling_flow.py::_validate_straight_flow_model`), so
the two families cannot be mixed silently.

## 4. Conditional recovery formulation (paper Sec. III-A, Eqs. (3)--(6))

Paper condition vector (Eq. (3)):
```
c_{k,n} = [m_n; qhat(t_n); v; sin psi; cos psi; ftilde_k; ttilde_n]
ftilde_k = 2k/(N_f-1) - 1,   ttilde_n = 2n/(N_s-1) - 1.
```
Pilot projectors (Eq. (4)):
```
Q = P P^dagger,  Q_perp = I - Q,  H_R = H Q,  H_N = H Q_perp
```
and the conditional MAP objective (Eq. (6)):
```
Hhat_MAP in argmin_H F(H) + R_CKM(H),
F(H) = ||H P - Y||_F^2 / (2 nu),   R_CKM(H) = -1/2 log p_CKM(H).
```

| Paper element | Code |
| --- | --- |
| normalized coordinates $\widetilde f_k,\widetilde t_n$ | `src/ckmsrf/data/lazy_dataset.py::CKMSRFDataset.__getitem__` (`tf = [2k/(Nf-1)-1, 2t/(Ns-1)-1]`) |
| UE state part of $\mat c$ | same function: `state = [observed_position(3); velocity(3); sin(yaw); cos(yaw)]` (8-dim) |
| prior condition $\mat c$ plumbing | `condition = {"scalar","adp","tx_beam","rx_beam","state","time","frequency"}`; consumed by `src/ckmsrf/losses/gmflow.py::extract_condition` |
| projectors $Q=\mat P\mat P^\dagger$ | `src/ckmsrf/physics/rc.py::build_physics_cache` (`projection = P @ pinv(P)`) |
| $H_R=HQ$ / $H_N=HQ_\perp$ | `src/ckmsrf/physics/guidance.py::apply_range_projection`, `::apply_null_projection` |
| data fidelity $F(\mat H)$ | `cache["M"] = P P^H / sigma2`, `cache["R"] = Y P^H / sigma2` in `build_physics_cache` |
| subspace error metric $E_{\mat A}$ | `src/ckmsrf/subspace_metrics.py::pilot_subspace_errors` (`e_obs`, `e_null`, closure check $E_{obs}+E_{null}=NMSE$) |

`R_CKM` is **not** evaluated as a density anywhere: the prior is represented by
the learned flow operator used inside PPC (Section 6). Eqs. (5)--(6) are the
conceptual Bayesian statement that motivates the splitting.
`Remark 1` (pilot-null components are invisible to $F$) is reflected in the
subspace metric above: `pilot_subspace_errors` projects the *same* error onto
`range(P)` and its orthogonal complement.

## 5. Knowledge-and-state conditioning (paper Sec. III-B, Eqs. (7)--(11))

Paper conditional velocity field and objective:
```
d phi_s / ds = v_theta(phi_s, s, c),  phi_0 = eps
L_FM(theta) = E_{pi,s} || v_theta(x_s, s, c) - (h - eps) ||^2
```
Modulation (Eq. (11)):
```
e_* = e_flow(s) + W_c e,    (gamma_l, beta_l) = MLP_l(e_*),
ztilde_l = SiLU[(1+gamma_l) * GN(Conv(z_l)) + beta_l],   e = phi_eta(c) in R^128.
```

| Paper element | Code | Key names |
| --- | --- | --- |
| $\mat c\mapsto\mat e=\phi_\eta(\mat c)\in\mathbb R^{128}$ | `src/ckmsrf/models/condition.py::CKMConditionEncoder` | `scalar_dim=6`, `state_dim=8`, `tx_beam_dim=64`, `rx_beam_dim=16`, `adp_shape=(16,16,12)`, `embed_dim=128`; five branches + `self.fusion` |
| scalar branch ($s_g$) | `::CKMConditionEncoder.forward` | `scalar_encoder = _VectorBranch(6, 64)` |
| state branch | same | `state_encoder = _VectorBranch(8, 64)`; `_state_value()` joins position/velocity/orientation |
| beam branch ($b^{tx},b^{rx}$) | same | `beam_encoder = _VectorBranch(64+16, 64)` |
| angular--delay branch ($\mat A_g$) | same | `adp_encoder` = Conv2d/GroupNorm/SiLU/Conv2d/AdaptiveAvgPool2d |
| time--frequency branch | same | `FourierPairEmbedding(fourier_bands=8)` + `tf_encoder` |
| $e_{flow}(s)$ | `src/ckmsrf/models/unet.py::ConditionalGMUNet.forward` | `time_mlp` = `SinusoidalPosEmb` + Linear/GELU/Linear, applied to `t*1000` |
| $W_c$ | same | `condition_projection = nn.Linear(condition_embed_dim, time_dim)` (zero-initialized) |
| $e_*=e_{flow}(s)+W_c e$ | same | `embedding = time_embedding + condition_projection(condition_embedding * keep)` |
| $(\gamma_l,\beta_l)=\mathrm{MLP}_l(e_*)$ | `src/ckmsrf/models/unet.py::ResnetBlock` | `self.mlp = Sequential(SiLU(), Linear(time_emb_dim, dim_out*2))`; chunked into scale/shift |
| $\widetilde z_l=\mathrm{SiLU}[(1+\gamma_l)\odot GN(Conv(z_l))+\beta_l]$ | `src/ckmsrf/models/unet.py::Block.forward` | `x = norm(proj(x)); x = x*(scale+1) + shift; return SiLU(x)` |
| U-Net widths / depth | `src/ckmsrf/models/unet.py::GMUNet` | `dim=32`, `dim_mults=(1,2,4)` $\Rightarrow$ widths (32,64,128); `num_res_blocks=2` (2 encoder + 3 decoder blocks per scale); skip concatenation in `_forward_with_embedding` |
| two output channels (real/imag velocity) | `::ConditionalStraightFlowUNet` | `out_channels=2`, `output_parameterization="straight_velocity"` |
| unconditional control ($U$) | `::ConditionalGMUNet.forward` with `force_unconditional=True` | set by `main/configs/standard/no_ckm.yaml` (`model.condition.force_unconditional: true`); the full encoder is kept so that architecture and parameter count match |
| condition dropout during training | `::ConditionalGMUNet.forward` | `cond_dropout` (`dropout_prob: 0.1`) |

The network is shared across locations; the 4.01 M parameter count quoted in
the paper (Sec. IV-A) is the count of this backbone. The released checkpoints
contain 4,010,818 parameters (recorded in [`docs/CHECKPOINTS.md`](CHECKPOINTS.md);
download manifest: `checkpoints/MANIFEST.json`).

## 6. FlowADMM / pilot--prior consensus (paper Sec. III-C, Eqs. (12)--(18), Algorithm 1)

### 6.1 Splitting and formal updates

Paper splitting (Eqs. (12)--(14)): copies $\mat H$ (data) and $\mat Z$
(prior), constraint $\mat H=\mat Z$, scaled dual $\mat U$, inverse penalty
$\tau$:
```
L_tau(H,Z,U) = F(H) + R_CKM(Z)
             + ||H - Z + U||_F^2 / (2 tau) - ||U||_F^2 / (2 tau)
H^{j+1} = argmin_H { F(H) + ||H - B^j||_F^2/(2 tau) },   B^j = Z^j - U^j
Z^{j+1} = argmin_Z { R_CKM(Z) + ||Z - W^{j+1}||_F^2/(2 tau) },  W^{j+1} = H^{j+1} + U^j
U^{j+1} = U^j + H^{j+1} - Z^{j+1}
```
Implementation: `src/ckmsrf/sampling_flow.py::sample_flow_admm`
(scaled dual is `scaled_dual`; the definition matches the paper because the
scaled form is used, `U^{j+1} = W^{j+1} - Z^{j+1}`).

### 6.2 Closed-form pilot correction (paper Eq. (15))

Paper:
```
H^{j+1} = B^j + (Y - B^j P) (P^H P + (nu/tau) I)^{-1} P^H
```
The code solves the equivalent Gaussian-posterior form
```
H^{j+1} = (P P^H / sigma2 + I / tau)^{-1} (Y P^H / sigma2 + B^j / tau)
```
by a linear solve; the two expressions agree after the Woodbury identity
(verified numerically to machine precision). The inverse is only
$N_p\times N_p$ in the paper form and $N_t\times N_t$ in the code form; this is
an implementation choice, not a modeling difference.

| Paper element | Code |
| --- | --- |
| exact data subproblem, Eq. (15) | `src/ckmsrf/physics/pnp.py::pnp_data_step` with `rho = 1/tau` |
| Gaussian-posterior linear solve | `src/ckmsrf/physics/rc.py::rc_proximal_update` (`precision = M + I/prior_var`, `rhs = R + prior_mean/prior_var`, `torch.linalg.solve`) |
| $\mat B^j=\mat Z^j-\mat U^j$ | `sample_flow_admm`: `pnp_data_step(z - scaled_dual, cache, rho=1.0/tau)` |
| preprocessing $P P^H$, $Y P^H$, $P^\dagger$, $P P^\dagger$ | `src/ckmsrf/physics/rc.py::build_physics_cache` |
| $\beta=\tau/(\tau+\nu)$ (semi-unitary analysis) | *analytic quantity only*; the code forms no explicit $\beta$. It is the diagonal weight implied by the closed form when $P^H P=I$. |

### 6.3 Conditional prior refinement and dual feedback (paper Eqs. (16)--(18))

Paper: the prior step is replaced by the learned flow operator with $M$
endpoints,
```
Shat^{(M)}_{s,CKM}(b) = (1/M) sum_m D_{s,c}(s b + (1-s) eps_m)
Z^{j+1} = T^{-1} Shat^{(M)}_{s_j,CKM}(T(W^{j+1}))
s_j = s_min + ((j+1)/K)^gamma (s_max - s_min)
U^{j+1} = W^{j+1} - Z^{j+1},   B^{j+1} = Z^{j+1} - U^{j+1}
```

| Paper element | Code | Key names |
| --- | --- | --- |
| $\mat W^{j+1}=\mat H^{j+1}+\mat U^j$ | `sample_flow_admm` | `x + scaled_dual` |
| endpoint average, Eq. (17) | `src/ckmsrf/sampling_flow.py::_average_renoise_denoise` | `latent_samples=M`; `path_state = t*reference + (1-t)*latent_std*latent`; estimate `= mean(path_state + (1-t)*velocity)` |
| flow-time schedule, Eq. (18) | `sample_flow_admm` | `fraction = ((iteration+1)/iterations)**gamma`; `time_value = time_min + fraction*(time_max-time_min)` |
| dual feedback | `sample_flow_admm` | `scaled_dual = scaled_dual + x - z` |
| initialization $\mat Z^0=\mat Y\mat P^\dagger$, $\mat U^0=0$ | `src/ckmsrf/sampling_flow.py::_prepare_problem` | `initialization="ls"` $\Rightarrow$ `torch.bmm(observation, p_pinv)`; `z = initial.clone()`, `scaled_dual = zeros_like(initial)` |
| $KM$ network evaluations, output $\widehat{\mat H}=\mat Z^K$ | `sample_flow_admm` | `diagnostics["denoiser_nfe"] = sum(sample_plan)`; returns `z` |
| optional three-phase $N_k$ plan | `src/ckmsrf/sampling_flow.py::build_flowadmm_sample_plan` | `k_avg_schedule="constant"` (paper) or `"three_phase"` (research option, not used for the paper table) |

Algorithm 1 of the paper maps line-by-line to the loop at
`src/ckmsrf/sampling_flow.py::sample_flow_admm` (lines ~447--476): input/setup,
`for j in range(K)`, schedule, data step, prior step, dual update, output.
The return value is `Z`, not `H`; with a perfect prior the two coincide.

## 7. PnP-Flow (comparison solver, paper Sec. IV-A)

Paper: explicit data gradient with step
`gamma_k = eta (1 - t_k)^alpha / L_F`, `L_F = lambda_max(P P^H)/nu`, prior
update by renoise--denoise, schedule `s_j = 0.2 + 0.8 j/K`, one MC sample per
update by default.

| Paper element | Code |
| --- | --- |
| gradient step and step size | `src/ckmsrf/sampling_flow.py::sample_pnp_flow` (`raw_gradient = (H P - Y) P^H / sigma2`, `gamma = step_scale * (1-t)**alpha / (lambda_max(P P^H)/sigma2)`) |
| prior update with $M$ samples | `::_average_renoise_denoise` (shared with KAP-Flow) |
| schedule $0.2+0.8j/K$ | `sample_pnp_flow` (`time_value = time_min + (iteration/iterations)*(time_max-time_min)` with `time_min=0.2`, `time_max=1.0`) |
| parameters $(\eta,\alpha,K,M)$ | `main/configs/standard/*.yaml`: `pnp_flow_step_scale: 1.5`, `pnp_flow_alpha: 0.25`, `pnp_iterations: 40`, `pnp_flow_latent_samples: 1` |
| shared prior | KAP-Flow and PnP-Flow load the **same** trained straight-flow checkpoint per condition (`flowadmm,pnpflow` in `configs/checkpoints.json`) |

## 8. Baselines shipped with the release

| Paper row | Code method / sampler | Main implementation |
| --- | --- | --- |
| LS | `ls` | `src/ckmsrf/baselines/classical.py::least_squares` |
| LMMSE | `lmmse` (needs fitted statistics) | `src/ckmsrf/baselines/classical.py::gaussian_lmmse`, `::select_statistics`; statistics from `src/ckmsrf/fit_channel_stats.py` |
| RC-U / RC-C / RC-S | `rcflow` / `srf` (sampler name `srf-rc`) | `src/ckmsrf/sampling.py::sample_ckmsrf` (GMFlow additive path + RC proximal update; optional PBG) |
| CM-U / CM-C | `pnp-cm` | `src/ckmsrf/sampling_cm.py::sample_ckmcm_pnp` (4 outer steps, rho 1, lambda 0.1) |
| PF-U / PF-C | `pnpflow` | Section 7 |
| KAP-U / KAP-C | `flowadmm` | Section 6 |

Also present in the code but **not** part of the paper's main comparison:
PnP-DM (`src/ckmsrf/sampling_pnpdm.py`), the diffusion/consistency/
additive-flow priors (`models/consistency.py`, `models/flow_matching.py`,
`losses/diffusion.py`, `losses/flow_matching.py`, `losses/gmflow.py`) and the
robustness track (`condition_corruption.py`, `models/robust_straight_flow.py`).

## 9. Paper hyperparameters and protocol (paper Sec. IV-A)

| Quantity | Paper value | Code / config |
| --- | --- | --- |
| $(K,\tau,s_{min},s_{max},\gamma)$ KAP | $(40,0.125,0,0.95,0.25)$ | `main/configs/standard/ckm.yaml` (`pnp_iterations`, `flow_admm_*`) |
| $M$ KAP | 1 (default), 1/2/4/8/16 in budget study | `flow_admm_k_avg`, CLI `--m` |
| $(K,\eta,\alpha)$ PF | $(40,1.5,0.25)$ | `main/configs/standard/*.yaml` (`pnp_flow_*`) |
| PF schedule $(s_{min},s_{max})$ | $(0.2,1.0)$ | `pnp_flow_time_min/max` |
| initialization | LS for both flow solvers | `flow_initialization: ls` |
| latent std | 1 | `flow_latent_std: 1.0` |
| flow training path | straight, $s\in[0,1]$, mini-batch OT | `straight_flow_matching: {time_min: 0, time_max: 1, coupling: minibatch_ot}` |
| backbone | dim 32, mults (1,2,4), cond embed 128 | `model.*` |
| optimizer / schedule | AdamW $2\times10^{-4}$, cosine warmup 1000 | `optimizer.*`, `scheduler.*` |
| EMA | 0.999 | `training.ema_decay` |
| train/val/test records | 10000/1000/2000 per site | see `docs/DATA.md` |
| test pair $(k,n)$ | $(23,3)$ zero-based | dataset `fixed` sampling default = the center pair $((N_f-1)//2,(N_s-1)//2)=(23,3)$ for $(N_f,N_s)=(48,8)$ (`src/ckmsrf/data/lazy_dataset.py::CKMSRFDataset.__init__`); the timing CLI exposes the same point as `--subcarrier 23 --snapshot 3` (`timing/benchmark_all.py`) |
| SNR grid | $\{-10,-5,\ldots,30\}$ dB | CLI `--snr-db` |
| inference seeds (standard flow tracks) | 3 seeds, `10009102/3/4` | `src/journal_flow/cli.py::make_jobs` |

The evaluation/noise protocol (three sites, map grids, pilot matrices,
$x$ dB $\nu$ formula, aggregate NMSE definition) is documented in
`docs/DATA.md` and `docs/RESULTS.md`; the checkpoint registry is in
[`docs/CHECKPOINTS.md`](CHECKPOINTS.md) and `configs/checkpoints.json`.

## 10. Chinese summary (中文要点)

- **KSC**：把历史 CKM 插值描述子、UE 状态（位置/速度/航向 sin/cos）、时频坐标
  拼成条件 $\mat c$，经 `CKMConditionEncoder` 融合为 128 维嵌入，再用 FiLM/AdaGN
  调制共享的 straight-flow U-Net（`models/condition.py` + `models/unet.py`）。
- **流匹配**：训练路径为 $x_t=(1-t)\epsilon+t\mat h$，回归速度
  $\mat h-\epsilon$，mini-batch OT 只置换潜变量；单步端点预测
  $D_t(x)=x+(1-t)v_\theta$ 即 `denoise()`。
- **PPC / FlowADMM**：$\mat H$（数据）与 $\mat Z$（先验）拆分，数据步是闭式
  噪声感知复数域解（`physics/rc.py::rc_proximal_update`，等价于论文
  $B+(\mat Y-\mat B\mat P)(\mat P^{\mathrm H}\mat P+\nu/\tau\mat I)^{-1}\mat P^{\mathrm H}$），
  先验步为 $M$ 端点平均的 flow 预测（`_average_renoise_denoise`），
  双反馈 $\mat U$ 累积两者差异；$K$ 轮输出 $\mat Z^K$。
- **命名**：论文 KAP-Flow = 代码 `flowadmm`（旧名 FlowADMM/FA），PnP-Flow =
  `pnpflow`（PF），RC/SRF = `rcflow`/`srf`，PnP-CM = `pnp-cm`；
  详细映射见 `docs/NAMING.md`。
- **代码与论文的差异**：论文中的 $R_{\rm CKM}$ 密度、$\beta=\tau/(\tau+\nu)$
  与后验积分公式是分析性表述，代码不显式计算；实现以 `sampling_flow.py` 的
  迭代为准。不确定处一律以 "see code" 标注，未做推断。

## 11. Where to look next

* `docs/NAMING.md` -- paper/code/legacy names, configs, checkpoints, table rows.
* `docs/RESULTS.md` -- all reported numbers with their source files.
* `results/table1.md` -- Markdown version of the paper's Table I.
* `docs/DATA.md` -- data layout, observation contract, CKM generator.
* `docs/CHECKPOINTS.md` -- released weights and how to place them.
* `CITATION.cff`, `citation.bib` -- provisional citation metadata (venue TBD).
