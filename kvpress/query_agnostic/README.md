# 不看未来问题的分块 KV 实验运行层

对应工作区的 `kv-query-agnostic-experiment-plan-2026-11-12.md`。本模块以独立运行层实现真正的逐块前向，复用 Qwen2 的投影、RoPE、输出投影和其余模型层。进入上下文时只替换该模型实例的 attention forward，退出时恢复。

从服务器环境准备到首次评分的执行顺序见本文末尾的[分阶段运行命令](#分阶段运行命令)。下面涉及的安装、下载和正式校准命令均留待服务器上执行；当前不连接服务器、不安装本机依赖、不下载主模型权重。

## 已实现

- 默认每次读入 2048 个正文 token；每层当前块完成 attention 后，对历史 KV＋新块做选择。
- 全局固定预算、静态 HM head 份额、保护下限、可用长度上限、饱和后的预算重分配和最大余数整数化。
- 每层每 KV head 独立的紧凑 K/V 数组；同步维护原始位置、来源块及持久分数。选择后按原始位置排序，无 head padding。
- `full`、`ea_once`、`keydiff`、`leverage`、`qfilters`、`fastkvzip`、`probe_mean`、`probe_max`、`fixed_random`。
- Leverage 的固定 Gaussian sketch、按原始位置逆 RoPE、FP32 ridge Cholesky；静态方法只对新 token 算分。
- 正文 Q 的有界加权聚类、真实 Q 代表、历史数量/SSE 合并；Mean 按数量加权，Max 不按数量加权。评分 softmax 分 tile 归约。
- EA 累计正文 Q 的 FP32 均值/协方差，在全文结束后按原始末位置旋转并压缩一次。
- 全部问题从冻结的正文缓存独立回答；只持久保存该题新增 KV，正文缓存不复制、不裁剪、不更新探针。异常也清理问题分支。
- 五项 SCBench 解析、字面模板中立化、独立 token 段、完整原始选项、正文 token 到字符位置、文件/输入/代码 hash。
- 每题断点文件、逐正文完成文件、失败记录、最终幸存位置、可选来源块保留日志和显存峰值。
- 调用原 KVzip/RepoQA 评分源码；空输出保留在分母，按文档、任务等权汇总。缺文档时不生成完整实验总分。

`budget` 表示等效每个 head 的条目数：全局目标为 `layers × kv_heads × budget`。模板前缀和正文 sink 包含在预算内。默认 recent window 为 0，问答阶段不再驱逐。

目前只接入单设备、batch=1、无 sliding window、无 RoPE scaling 的 `Qwen2ForCausalLM`。通过实验 extra 固定 Transformers 4.57.6；模型并行和 CPU/disk offload 尚未接入。

## 本地验证

不需要下载任何模型权重，随机小 Qwen2 即可跑通九种路径：

```bash
python -m kvpress.query_agnostic.smoke --device cpu
python -m kvpress.query_agnostic.smoke --device cuda --dtype bfloat16
python -m kvpress.query_agnostic.smoke --device cuda --dtype bfloat16 --backend flash_varlen
python -m pytest tests/test_query_agnostic.py tests/test_query_agnostic_data.py -q
```

`smoke` 明确使用随机主模型、随机 filter/gate 和开发用预算，输出不能作为方法质量结果。测试中的 CUDA/FlashAttention 项会根据当前进程的设备与扩展可见性自动启用。

本机已在 `pytorch_env` 的 PyTorch 2.5.1、CUDA runtime 12.1、Transformers 4.57.6 下验证 CPU 与 RTX 4060 Laptop 8GB；FlashAttention 2.7.4.post1 的真实变长内核也已验证。沙箱内可能看不到 `/dev/nvidia*`，CUDA 验证需在设备可见的执行环境中运行。

## A100 上安装与下载

进入同步到服务器的 kvpress 目录，安装到实验专用环境：

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[query-agnostic]' 'torch==2.5.1'
uv pip install --no-build-isolation 'flash-attn==2.7.4.post1'
```

选择与驱动和 PyTorch 匹配的 CUDA 包/预编译 wheel；上面的 FlashAttention 命令可能触发本地编译。正式对比固定同一软件环境与 attention 后端。

模型下载不需要 GPU。把以下路径换成租用平台的**持久盘**路径，可以先用 CPU 实例准备：

```bash
python -m kvpress.query_agnostic.download --output /persistent/models/Qwen2.5-7B-Instruct-1M
```

该命令固定 checkpoint revision 为 `e28526f7bb80e2a9c8af03b831a9af3812f18fba`，记录下载文件 hash。运行器接收本地路径时会验证 `experiment_snapshot.json`。`--tokenizer-only` 可先下载配置和 tokenizer；正式运行会拒绝只有 tokenizer 的快照。

下载脚本不会自动启动实验。仓库也没有携带主模型权重或代替离线校准的假数据。

## 离线产物约定

HM、Q-Filters、Fast KVzip gate 必须来自计划规定的独立校准/训练文本及目标 checkpoint。三条离线生成流水线已经接入，关键定义和与作者实现的差异见 [源码核对记录](SOURCE_AUDIT.md)。HM 预算直接调用作者 checkout 的 `document_shares` 和 `solve_scale`。

每个文件旁放同名 `.json` 元数据，例如 `hm_profiles.npz` 对应 `hm_profiles.json`：

```json
{
  "kind": "hm",
  "model_id": "Qwen/Qwen2.5-7B-Instruct-1M",
  "model_revision": "e28526f7bb80e2a9c8af03b831a9af3812f18fba",
  "calibration_manifest_sha256": "校准文本清单的 SHA256",
  "artifact_sha256": "产物文件的 SHA256"
}
```

文件格式：

| 产物 | 内容 | kind |
|---|---|---|
| `hm_profiles.npz` | `shares_4096`、`shares_8192`、`shares_16384`、`shares_32768`，各为正值 `[28,4]` 静态份额 | `hm` |
| `qfilters.safetensors` | `q_filters`，形状 `[28,4,128]`，沿用原实现负号约定 | `qfilters` |
| `gates.pt` | `{"layers": [gate.state_dict(), ...]}`，共 28 层，可由 `weights_only=True` 加载 | `fastkvzip` |

每个产物的文本来源、训练参数和验证切分应登记在其校准清单内。运行配置还会独立记录产物及元数据的文件 hash。

## 生成离线产物

先同步本仓库及 `marginal-attention` 作者 checkout 到服务器。模型已按上文下载后，可以执行完整流水线：

```bash
python -m kvpress.query_agnostic.calibrate all \
  --work-dir /persistent/calibration/qwen25-7b-1m \
  --model /persistent/models/Qwen2.5-7B-Instruct-1M \
  --author-repo ../marginal-attention \
  --device cuda --backend flash_varlen
```

首次运行会解析一次 FineWeb-Edu `sample-10BT` 的数据 revision，再固定该 revision 流式读取。可通过 `--dataset-revision` 指定 commit；也可用 `--source-jsonl` 传入已冻结的本地来源池，每行含 `text` 和可选唯一 `id`，记录文件 hash。语料不足会报错，不缩减样本清单。

默认语料为 HM 200×16,384、Q-Filters 20×2048、gate 训练短样本 500K＋长样本 500K token，额外验证短样本 50K＋长样本 100K。每个源文档只分配给一个窗口；重复 ID/文本去重。长 gate 窗口允许拼接多个来源，源边界保存在 manifest。HM/Q-Filters/训练/验证不共用来源。

各阶段也可单独运行。以下命令均需传相同的 `--work-dir`；收集阶段还需指定 `--model`、`--device` 和 `--backend`：

| stage | 工作 | 输出目录 |
|---|---|---|
| `prepare` | 冻结语料、token IDs 与来源清单；不载入模型权重 | `tokens/`、`calibration_manifest.json` |
| `hm-labels` | Full prefill、512 重建块、完整 softmax KVzip+ 标签 | `hm_labels/` |
| `hm-build` | 调用作者 quantile 与 floor/cap 求解器；CPU 可执行 | `artifacts/hm_profiles.npz` |
| `qfilters-collect` | 捕获 post-RoPE Q、固定位置抽样 | `qfilter_samples/` |
| `qfilters-build` | 未中心化 SVD、符号与 GQA 均值 | `artifacts/qfilters.safetensors` |
| `gate-collect` | 原始 KVzip 标签与逐层 BF16 mmap 特征 | `gate_features/` |
| `gate-train` | 释放主模型，逐层训练 gate、验证、断点恢复 | `artifacts/gates/gates.pt` |

`all` 会在 gate 训练前释放主模型。收集阶段按完整文档恢复；gate 每 500 步保存 optimizer/RNG/排序状态，已完成的层可直接复用。改变代码、参数、来源、设备或依赖时使用新工作目录。同一个目录只运行一个写入进程。

默认标签评分 tile 为 128 Query 行×4096 Keys；`--score-query-tile`、`--score-key-tile` 可调整工作区。prefill 默认 2048，可显式指定 `--prefill-chunk`。这些设置进入配置 hash。HM 与 gate 使用不同标签定义，不能交换二者的 label store。

使用随机小模型验证整条离线链路，不下载 7B 权重：

```bash
python -m kvpress.query_agnostic.calibration_smoke \
  --work-dir /tmp/kv-calibration-smoke --author-repo ../marginal-attention
python -m kvpress.query_agnostic.calibration_smoke \
  --work-dir /tmp/kv-calibration-smoke-cuda --author-repo ../marginal-attention \
  --device cuda --backend flash_varlen
```

随机模型产物带 `synthetic/` 模型身份，正式运行入口会拒绝。正式 7B 的校准清单、标签和训练产物仍需要在服务器实际生成。

## 先从参考 4× 开始

保留原方案的固定容量协议，可以按 **32K → 16K → 8K → 4K** 逐档试跑；在 128K 参考长度下分别为 4×、8×、16×、32×。32K 是新增的低压缩强度开发档，原默认 8K 不变。

**4K 容量与 4× 压缩不同。** 实际压缩倍数约为 `正文长度 / 固定预算`。32K 缓存对 64K 正文约为 2×，对 128K 正文才约为 4×；报告中记录真实倍数。如果给每条正文预先设 `B=N/4`，就引入了最终长度信息，应另列为已知长度协议。

连接服务器后建议按以下顺序推进：环境与随机小模型检查 → 下载固定模型 → 7B 短输入数值检查 → 每任务一条完整正文的 Full/显式均匀预算开发验证 → HM 校准与构建 → 各方法 32K 的固定开发清单 → 冻结质量判定标准后逐档增加压缩。Q-Filters 校准和 Fast KVzip 特征/训练可分别执行，尚未得到 gate 时，其余方法可以先运行。

HM 标签只需生成一次，`hm-build` 同时导出四档 profile。Q-filter 和 gate 跨预算复用。每个预算的在线方法需要从原文重新 prefill；把 32K 的最终缓存再裁成 16K，不能替代从头使用 16K 预算的在线路径。Full 结果在不同预算之间复用。

32K 的 BF16 纯 KV 为 **1.75 GiB**，8K 为 **0.4375 GiB**，额外约 **1.31 GiB**；更多历史 KV 也会增加 attention 和选择器耗时。离线标签/特征阶段和 Full 的峰值不会因该档位而减少，A100 上仍需先做完整正文 profiling。

晋档由固定开发清单上的相对 Full 得分及逐任务表现判断，当前不自动依据未知阈值晋档。正式表格如实标注未运行的档位；若要做同预算方法比较，使用相同样本和预算，不能只汇总过线的方法。

运行 32K 档时，在下方命令中将 `--budget 8192` 改为 `--budget 32768`；每个方法/预算使用独立输出目录。

## 先验证一条完整正文

从 kvpress 仓库根目录执行，替换模型、数据和产物路径：

```bash
python -m kvpress.query_agnostic.run \
  --model /persistent/models/Qwen2.5-7B-Instruct-1M \
  --dataset-dir ../data/kv_query_agnostic_2026-11-10/dataset_samples \
  --method keydiff --budget 8192 \
  --hm-profile /persistent/calibration/qwen25-7b-1m/artifacts/hm_profiles.npz \
  --backend flash_varlen --device cuda \
  --tasks scbench_qa_eng --limit 1 \
  --output results/query_agnostic/keydiff-8192-validation
```

`--limit 1` 是每项任务取一条完整正文，不截断正文。正式运行去掉此参数。

`--method full` 无需 HM。其余方法缺少 HM 时默认报错。开发或均匀预算消融可明确指定 `--allocation uniform`，该选择会写入配置。Q-Filters 额外传 `--filters`；Fast KVzip 额外传 `--gates`。可用选项见 `python -m kvpress.query_agnostic.run --help`。

相同命令可恢复未完成的目录；恢复仍需重新 prefill 该正文，已经落盘的题目可跳过生成。配置、输入文件、产物或执行代码 hash 变化时，应使用新目录。每个输出目录只运行一个写入进程。

## 输出与后续工作

输出包含 `config.json`、`environment.json`、`data_manifest.jsonl`、`documents/`、`tokens/`、`final_survivors/` 和 `failures.json`。逐正文 JSON 包含原始预测、参考答案、时延、真实 KV/辅助状态字节和块级记录。`--diagnostics` 加入逐层/head/来源块的候选与保留计数；模板来源为 -1，正文从 0 开始。

安装评分依赖并调用计划指定的原评分源码：

```bash
uv pip install -e '.[query-agnostic-eval]'
python -m kvpress.query_agnostic.score \
  --run results/query_agnostic/keydiff-8192-validation \
  --metric-root ../data/kv_query_agnostic_2026-11-10/repos/KVzip
```

结果保存为 `scores.json`，登记原评分文件 hash。VT、QA、选择题、Summary 调用原 `evaluate_answer`；RepoQA 调用原 `needle_evaluator` 并保留所有空输出，使用相同的 0.8 相似度阈值。

当前完成的是执行底座、方法路径、评分入口和三条离线产物流水线。attention/evidence 深度诊断、冻结编码 replay、状态内存匹配、bootstrap 与最终图表尚未实现。已有日志不是完整的图 A/B/C/D 数据集。

尚未在 7B 主模型和 A100 上做 64K–128K 验证，方案里的显存范围仍是估算。A100 就绪后先跑 Full 与 KeyDiff 的一条完整正文，确认数值、峰值和耗时，再跑含校准产物的主实验。聚类目前逐 Query head 处理，长文档的开销需实测后再优化。

## 分阶段运行命令

以下均在未来的服务器上执行。先同步**当前工作区的完整代码，包含新增文件**，以及作者源码、五项 parquet 和原评分源码。只拉取 NVIDIA 上游仓库不会得到本次新增实现。目录关系保持为：

```text
llm_study/
├── kvpress/                         # 本次修改后的项目
├── marginal-attention/              # 作者 checkout，保留 .git，用于登记 commit
└── data/kv_query_agnostic_2026-11-10/
    ├── dataset_samples/            # 五项 SCBench parquet
    └── repos/KVzip/results/         # metric.py、repo_qa_utils.py 与必要依赖文件
```

### 1. 环境和固定路径

进入服务器上的 `kvpress` 目录，将 `/persistent` 换成实际持久盘路径。以下命令放在同一个 Bash 会话中执行：

```bash
nvidia-smi
free -h
df -h
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[query-agnostic,query-agnostic-eval]' 'torch==2.5.1'
uv pip install --no-build-isolation 'flash-attn==2.7.4.post1'

export KV_MODEL_DIR=/persistent/models/Qwen2.5-7B-Instruct-1M
export KV_CALIB_DIR=/persistent/calibration/qwen25-7b-1m
export KV_RESULT_DIR=/persistent/results/query-agnostic
export KV_DATA_DIR="$(pwd)/../data/kv_query_agnostic_2026-11-10/dataset_samples"
export KV_AUTHOR_DIR="$(pwd)/../marginal-attention"
export KV_METRIC_DIR="$(pwd)/../data/kv_query_agnostic_2026-11-10/repos/KVzip"
```

安装时匹配服务器的驱动、CUDA/PyTorch 和 FlashAttention wheel；当前验证过的是 PyTorch 2.5.1、Transformers 4.57.6、FlashAttention 2.7.4.post1。A100 的实际环境仍需检查。先执行不依赖权重的小模型检查，再下载主模型：

```bash
python -m kvpress.query_agnostic.smoke --device cuda --dtype bfloat16 --backend flash_varlen
python -m kvpress.query_agnostic.calibration_smoke \
  --work-dir "$KV_CALIB_DIR/synthetic-check" --author-repo "$KV_AUTHOR_DIR" \
  --device cuda --backend flash_varlen
python -m kvpress.query_agnostic.download --output "$KV_MODEL_DIR"
```

`synthetic-check` 只存随机小模型产物，后续正式校准使用其上级 `$KV_CALIB_DIR`，两者不混用。模型下载可以提前在具有相同持久盘的 CPU 实例上完成。

### 2. 先跑 Full，检查完整正文与评分

`--limit 1` 表示五项任务各取一条完整正文。此步骤不需要 HM、filter 或 gate，也不代表已经通过完整 7B 验证。

```bash
python -m kvpress.query_agnostic.run \
  --model "$KV_MODEL_DIR" --dataset-dir "$KV_DATA_DIR" \
  --method full --backend flash_varlen --device cuda --limit 1 \
  --output "$KV_RESULT_DIR/pilot/full"
python -m kvpress.query_agnostic.score \
  --run "$KV_RESULT_DIR/pilot/full" --metric-root "$KV_METRIC_DIR"
```

查看 `scores.json`、逐题预测和正文级峰值/耗时。先排查 OOM、空输出、模板/评分异常，再启动耗时更长的校准或全量评测。尚未具备 HM 时，可额外显式使用 `--allocation uniform --method keydiff --budget 32768` 做工程验证，并放在独立的开发结果目录；该结果不标记为 HM。

### 3. 先生成 HM，再逐项生成 filter 和 gate

```bash
python -m kvpress.query_agnostic.calibrate prepare \
  --work-dir "$KV_CALIB_DIR" --model "$KV_MODEL_DIR"
python -m kvpress.query_agnostic.calibrate hm-labels \
  --work-dir "$KV_CALIB_DIR" --model "$KV_MODEL_DIR" --author-repo "$KV_AUTHOR_DIR" \
  --device cuda --backend flash_varlen
python -m kvpress.query_agnostic.calibrate hm-build \
  --work-dir "$KV_CALIB_DIR" --author-repo "$KV_AUTHOR_DIR" --device cpu
```

得到 `artifacts/hm_profiles.npz` 后，EA-once、KeyDiff、Leverage、ProbeMean/Max 即可运行。Q-Filters 的准备命令为：

```bash
python -m kvpress.query_agnostic.calibrate qfilters-collect \
  --work-dir "$KV_CALIB_DIR" --model "$KV_MODEL_DIR" --device cuda --backend flash_varlen
python -m kvpress.query_agnostic.calibrate qfilters-build \
  --work-dir "$KV_CALIB_DIR" --device cuda
```

Fast KVzip 可以随后准备；其他方法不必等 gate 训练完成：

```bash
python -m kvpress.query_agnostic.calibrate gate-collect \
  --work-dir "$KV_CALIB_DIR" --model "$KV_MODEL_DIR" --device cuda --backend flash_varlen
python -m kvpress.query_agnostic.calibrate gate-train \
  --work-dir "$KV_CALIB_DIR" --device cuda
```

需要一次完成所有阶段时可使用前文的 `calibrate all`。首次租机预跑建议分阶段执行，以便先测量真实耗时和工作区。

### 4. 各方法先跑 32K 固定容量

下面函数按方法启动独立进程，默认五项任务各一条完整正文，并立即评分。它不会自动根据分数晋档。

```bash
export KV_BUDGET=32768
run_pilot() {
  local KV_METHOD="$1"
  shift
  local KV_OUTPUT="$KV_RESULT_DIR/pilot/b$KV_BUDGET/$KV_METHOD"
  python -m kvpress.query_agnostic.run \
    --model "$KV_MODEL_DIR" --dataset-dir "$KV_DATA_DIR" \
    --method "$KV_METHOD" --budget "$KV_BUDGET" \
    --hm-profile "$KV_CALIB_DIR/artifacts/hm_profiles.npz" \
    --backend flash_varlen --device cuda --limit 1 --output "$KV_OUTPUT" "$@" || return
  python -m kvpress.query_agnostic.score --run "$KV_OUTPUT" --metric-root "$KV_METRIC_DIR"
}
for KV_METHOD in ea_once keydiff leverage probe_mean probe_max; do
  run_pilot "$KV_METHOD" || break
done
```

相应产物生成后，再执行：

```bash
run_pilot qfilters --filters "$KV_CALIB_DIR/artifacts/qfilters.safetensors"
run_pilot fastkvzip --gates "$KV_CALIB_DIR/artifacts/gates/gates.pt"
```

这份小清单用于工程和探索性判断，逐任务分数与 Full 同时查看，不用五条正文的分数直接声称方法优劣。若继续比较更强压缩，将 `KV_BUDGET` 依次改为 `16384`、`8192`、`4096`，重新调用需要运行的方法；每档从正文重新编码，输出进入不同目录。

### 5. 固定配置后执行全量

全量必须使用新输出目录；`--limit 0` 表示全部 286 条正文、1413 个问题。例如 KeyDiff 的 32K 档：

```bash
python -m kvpress.query_agnostic.run \
  --model "$KV_MODEL_DIR" --dataset-dir "$KV_DATA_DIR" \
  --method keydiff --budget 32768 \
  --hm-profile "$KV_CALIB_DIR/artifacts/hm_profiles.npz" \
  --backend flash_varlen --device cuda --limit 0 \
  --output "$KV_RESULT_DIR/main/b32768/keydiff"
python -m kvpress.query_agnostic.score \
  --run "$KV_RESULT_DIR/main/b32768/keydiff" --metric-root "$KV_METRIC_DIR"
```

其他方法替换 `--method`，并为 Q-Filters/Fast KVzip 添加上面的产物参数。全量 Full 也以 `--limit 0` 单独运行并评分一次，结果跨预算复用；预跑 Full 只有小清单，不能代替全量基线。

断线后重新激活环境、恢复路径变量，执行相同命令即可恢复。输出中 `complete=false` 或存在 `.error.json` 时查看错误并重跑，不能当成完整主实验结果。代码、参数、输入或环境变化时改用新目录。

### 交付边界

当前已具备从校准语料冻结到三类产物生成、在线压缩、逐题生成和原评分器汇总的执行链路。新增 32K 档用于较低压缩强度预跑；它对不同长度正文不是严格统一的 4×。

尚未完成的部分是深度 attention/evidence 诊断、固定编码 replay、总状态内存匹配、bootstrap 和最终图表。目标 7B/A100 的数值、峰值显存与耗时也仍需在上述首次运行时验证，当前通过的是小模型 CPU/CUDA 工程测试。
