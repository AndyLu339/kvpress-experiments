# 不看未来问题的分块 KV 实验运行层

对应工作区的 `kv-query-agnostic-experiment-plan-2026-11-12.md`。本模块以独立运行层实现真正的逐块前向，复用 Qwen2 的投影、RoPE、输出投影和其余模型层。进入上下文时只替换该模型实例的 attention forward，退出时恢复。

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

HM、Q-Filters、Fast KVzip gate 必须来自计划规定的独立校准/训练文本及目标 checkpoint。离线重建标签生成、HM 校准、filter 构建与 gate 训练流程仍需接入；本模块目前负责校验并消费它们。

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
| `hm_profiles.npz` | `shares_4096`、`shares_8192`、`shares_16384`，各为正值 `[28,4]` 静态份额 | `hm` |
| `qfilters.safetensors` | `q_filters`，形状 `[28,4,128]`，沿用原实现负号约定 | `qfilters` |
| `gates.pt` | `{"layers": [gate.state_dict(), ...]}`，共 28 层，可由 `weights_only=True` 加载 | `fastkvzip` |

每个产物的文本来源、训练参数和验证切分应登记在其校准清单内。运行配置还会独立记录产物及元数据的文件 hash。

## 先验证一条完整正文

从 kvpress 仓库根目录执行，替换模型、数据和产物路径：

```bash
python -m kvpress.query_agnostic.run \
  --model /persistent/models/Qwen2.5-7B-Instruct-1M \
  --dataset-dir ../data/kv_query_agnostic_2026-11-10/dataset_samples \
  --method keydiff --budget 8192 \
  --hm-profile /persistent/calibration/hm_profiles.npz \
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

当前完成的是执行底座、方法路径与评分入口。计划中的离线产物流水线、attention/evidence 深度诊断、冻结编码 replay、状态内存匹配、bootstrap 与最终图表尚未实现。已有日志不是完整的图 A/B/C/D 数据集。

尚未在 7B 主模型和 A100 上做 64K–128K 验证，方案里的显存范围仍是估算。A100 就绪后先跑 Full 与 KeyDiff 的一条完整正文，确认数值、峰值和耗时，再跑含校准产物的主实验。聚类目前逐 Query head 处理，长文档的开销需实测后再优化。
