# 离线流水线源码核对

核对日期：2026-11-12。实现位置为 `calibrate.py`、`reconstruction.py`、`calibration_ops.py`、`train_gates.py`。产物 JSON 记录实际输入、代码、模型 revision 和作者源码 hash。

## HM：复用作者预算函数，独立生成正确的 KVzip+ 标签

依据本地 `marginal-attention` **Git 已跟踪文件**，commit `63277cefb8c4ade6f892966e2b8144060a2c2e1a`。仓库内未跟踪的 `exp/kv/local_repro/` 等本地实验目录没有作为作者实现依据。

- [features.py](https://github.com/Flegyas/marginal-attention/blob/63277cefb8c4ade6f892966e2b8144060a2c2e1a/src/aspace/kv/features.py)：`hooked_passes`、`label_accumulator`、`kv_group_log`、`prefill_text`、`REPEAT_PROMPT`。
- [kvzip_shares.py](https://github.com/Flegyas/marginal-attention/blob/63277cefb8c4ade6f892966e2b8144060a2c2e1a/exp/kv/budget/kvzip_shares.py)：`document_shares`、`solve_scale`。
- [nemotron_shares.py](https://github.com/Flegyas/marginal-attention/blob/63277cefb8c4ade6f892966e2b8144060a2c2e1a/exp/kv/budget/nemotron_shares.py)：文档均值、label/head 轴顺序和元数据。
- [evaluate.py](https://github.com/Flegyas/marginal-attention/blob/63277cefb8c4ade6f892966e2b8144060a2c2e1a/exp/kv/evaluation/evaluate.py)：`build_quota_scorer` 解包 EA registry 实例，保留 `epsilon=1e-2`。

确认的语义：

1. HM 的 prefill 使用 user-only chat template，由 tokenizer 插入默认模板；正文和模板前缀均在完整缓存中。
2. 每个重建块前回滚到完整正文缓存。重新输入固定 repeat prompt，再 teacher-force 当前块；不携带上一块重建产生的 KV。作者默认重建块长 **512**。
3. attention 的 softmax 分母包含当时全部可见 K，包括完整正文、repeat prompt、当前重建副本的因果前缀。prompt 行也参与标签最大值。
4. 每个 Query head 上先计算 `max_query(attention / max(||hidden_input||,1e-6)) * ||W_O(head) v||`，再对同一 KV head 的 Query 组取最大值。存储 `log(max(score,1e-12))`。
5. `||W_O v||` 包含该 Query head 的输出投影，不能换成 `||v||`；也不能把 Fast KVzip 的局部 softmax 标签乘范数后当作 HM 标签。
6. 每文档在全部 layer/head/token 标签上做一次 `np.quantile`，比较符号为 **>=**。有 ties 时不改成强行选固定数量。
7. 对文档份额等权平均，再由作者 `solve_scale` 求 `clip(c*share,0.01,1)`，使平均保留率精确匹配目标。构建程序直接动态加载作者这两个函数，没有重写近似求解器。

与作者原实验的有意差异：按当前计划使用 200 条 FineWeb-Edu、每条 16,384 个正文 token；参考长度 131,072，对应保留率 1/32、1/16、1/8。作者使用 Nemotron 及 1/4、1/8、1/16。新的 HM profile 必须在目标 Qwen2.5 checkpoint 上重新测量。

分阶段试跑另增 32K 固定容量，参考保留率为 1/4；与原三档共用相同离线标签，独立计算该档的 quantile、文档均值和 floor/cap rescale。每条自然长度正文的实际压缩倍数按其 token 数与固定预算另行报告。

为了 A100 40G，prefill 分成 2048 块，标签 attention 以 Query/Key tile 计算；不分配完整 N×N attention。使用 FP32 softmax 归约，再转回 Q dtype，之后按作者公式取最大值。分块与 GEMM tile 可能产生浮点舍入差异，测试用作者原 `label_accumulator` 对相同 Q/K/V 做数值对照，不宣称 BF16 bitwise 相同。

## EA-once：作者 registry 参数已补齐

作者 HM+EA 的 scorer 是 registry 解包后的 `ExpectedAttentionPress(epsilon=1e-2,use_vnorm=True)`。因此本运行层补入可记录的 `ea_epsilon=1e-2`，评分为 `(softmax_group_mean + epsilon) * ||V||`。EA 此处使用 V 范数，区别于 HM 标签的 `||W_O V||`。

在线协议按实验计划累计**全部正文** Q 的 FP32 均值/协方差，排除模板 Q，未来旋转从原始完整末位置开始，窗口 512；保护模板和正文 sink。作者默认 `get_query_statistics` 在单次输入上跳过最初 `n_sink` 个 hidden states，这一统计取样规则与本实验的正文分段规则不同，不能把本协议描述为作者原始输入协议的逐字复现。测试固定同一组统计量后，与原 `ExpectedAttentionPress.score` 比较。

## Q-Filters：post-RoPE Q、未中心化 SVD、U 的多数符号

依据 `qfilters` commit `cda5c38d3aa3f11815cf991661be80cb4f345815` 的 `make_filters.py` 和 `modeling/modeling_qwen2.py`。

- 直接输入原文 token，**不加 chat template**。捕获含投影 bias 的 post-RoPE Q。
- 每层从全部 20×2048 个 token 位置中无放回抽 3000 个位置，所有 Query heads 使用相同的该层位置清单；清单和 seed 冻结。
- 未中心化 SVD。由第一列 U 中正数占比是否大于 1/2 决定符号；保存负号 filter，再对 GQA 组求均值；不在组均值后重新单位化。
- `full_matrices=False` 只减少无用的 U 列，不改变所需主方向。
- 作者脚本 `sample_count < num_sequences` 实际会少处理一条。本实现按计划处理恰好 20 条，并明确登记该修正。

## Fast KVzip：原始局部重建标签和训练/推理一致性

依据 [FastKVzip](https://github.com/Janghyun1230/FastKVzip/tree/e04afaa6007c58d18500d0698d2d8ae592ff22c2) commit `e04afaa6007c58d18500d0698d2d8ae592ff22c2`：`prefill/model/wrapper.py`、`prefill/model/template.py`、`prefill/attention/score.py`、`optim.py`、`utils.py`、`load_data.py`。

- 完整 prefill，不驱逐。训练特征是每层 attention 输入 hidden states，去掉模板前缀，与正文标签逐 token 对齐。
- 使用作者 Qwen 模板（包含其离线通用说明），重建块 **2000**。首块 repeat 提示，后续块提示加前一块末 8 个 token，再加聊天结束/assistant 起始段，最后 teacher-force 目标块。
- 每块临时 KV 用完回滚。实际前向能看完整正文；用于标签评分的局部 softmax 键集合仅为 **整个模板 prefix（作者的 sink）＋目标正文块＋本轮重建输入**，对重建输入应用因果 mask。
- 标签是局部 attention 在 Query 行和 GQA 组上的最大值，范围 [0,1]。不乘 `||V||`、`||W_O V||`，不除 hidden norm，不取对数。
- gate dimension 16、sink 参数数 16；soft-target BCE，先对 token 求均值，再对 KV heads 求和。SGD、momentum 0.9、lr 0.2、5000 更新步，学习率函数保留作者 warm-up 与后半程分段规则。
- 作者训练 `optim.Weight.forward` 把 b 加两次，推理 `prefill/attention/gate.py` 和 kvpress 的 gate 只加一次。本实现训练 `2*b` 参数化并保留相应梯度，导出时将 `2*b` 存入部署 gate 的 b，测试导出前后预测一致。

按实验计划的有意改动：短样本 10K–30K 共 500K token，长样本 100K 共 500K；长样本可由同一 split 的不同独立来源拼成，来源边界登记。额外验证集为 50K 短样本＋100K 长样本，按来源文档完全分离；不采用作者每第 100 个 token 留作验证的切分。batch 1024（作者代码默认 1000）。逐层训练以控制内存；各层独立初始化 seed，使用同一随机 token 排序规则，500 步做一次分批完整验证。BF16 特征按 uint16 原始位写 mmap，不转 FP16。

恢复点保存 gate、SGD momentum、训练步、当前 epoch 排序、CPU generator 状态和历史指标。已完成产物不会因重新打包 `torch.save` 改变文件 hash。开发用随机模型产物带 `synthetic/` 身份，正式加载器会拒绝它们。
