# 粤语 UD dependency parsing：数据准备与固定划分

本项目目前只完成数据获取、检查、固定划分和复现验证。没有加载模型、训练、计算 LAS、做数据增强，也没有加入儿童 SRep 数据。

## 数据版本与许可

采用 Universal Dependencies 的正式发布 tag `r2.18`（UD 2.18，2026-05-15），而不是浮动的默认分支。

| 数据 | 官方仓库 | tag | 完整 commit SHA | 原始 CoNLL-U SHA-256 | 许可 |
|---|---|---|---|---|---|
| Cantonese-HK | [UD_Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK/tree/r2.18) | `r2.18` | `fcc7dd5b5eb97004441c4aa20704055ddde7a748` | `cbd843a195d0db4cdafbf6fcafb7b7b559afea750411006f4728311e70cc4e2a` | CC BY-SA 4.0 |
| Chinese-HK（仅作平行映射证据，不进入训练） | [UD_Chinese-HK](https://github.com/UniversalDependencies/UD_Chinese-HK/tree/r2.18) | `r2.18` | `72dbb27668c13daa1fb456a14af823d445dc4c3a` | `a71ff348dd1e45c2cd525ed537a78d6c4b6c5edc15a19641bedc0a8465de820e` | CC BY-SA 4.0 |

原始文件、仓库 README 和许可证原文保存在：

- `data/raw/ud_cantonese_hk-r2.18/`
- `data/raw/ud_chinese_hk-r2.18/`

原始下载地址分别为 [粤语 CoNLL-U](https://github.com/UniversalDependencies/UD_Cantonese-HK/raw/r2.18/yue_hk-ud-test.conllu) 和 [普通话 CoNLL-U](https://github.com/UniversalDependencies/UD_Chinese-HK/raw/r2.18/zh_hk-ud-test.conllu)。许可见各原始目录内的 `LICENSE.txt`。

实际从文件计算得到 1,004 句、13,918 个整数 ID 词节点，与 r2.18 官方 README 的参考值相同；这两个数字并未被用作程序必须匹配的常量。

## 自定义实验划分

`UD_Cantonese-HK` 官方只提供 `yue_hk-ud-test.conllu`。本项目把该文件重新划成自己的 train/dev/test；**这是自定义实验划分，不是官方 UD 划分**。

固定参数为 seed 42，目标句数比例为 80%/10%/10%。脚本按 CoNLL-U 句块读取，以 NFC 规范化后的整数 ID 词节点 FORM 序列建立精确键，并以这些 FORM 无分隔拼接后的字符串建立第二个键。任一键相同即连边，连通分量作为不可拆分的 group。group 先按稳定 source key 排序，再由 Python `random.Random(42)` 打乱，按累计句数中点和 80%/90% 边界分配；各输出内恢复原语料顺序。

没有执行简繁转换、删标点或其他文本改写，也没有去重。原始注释、`sent_id`、`text`、全部十列、multiword token 行和 empty node 行均按完整句块原样输出。当前 r2.18 文件本身没有 multiword token 或 empty node 行。

实际结果：

| split | 句数 | 句数比例 | 整数 ID 词节点数 |
|---|---:|---:|---:|
| train | 803 | 79.980% | 11,001 |
| dev | 101 | 10.060% | 1,557 |
| test | 100 | 9.960% | 1,360 |

官方 README 可靠地用数字 `sent_id` 范围标明四个来源，因此本项目据此统计来源分布，不做猜测：

| 来源 | train | dev | test |
|---|---:|---:|---:|
| Missing days / 小時光 | 330 | 32 | 48 |
| Tempo in Temple / 廟眾樂樂 | 115 | 15 | 7 |
| What day is today / 今日星期幾 | 85 | 9 | 9 |
| Election of President（LegCo，2016-10-12） | 273 | 45 | 36 |

这是同一 treebank 内的 grouped random split，**不等于** leave-one-domain-out、跨领域测试或跨儿童泛化测试。来源比例也未作分层约束；尤其 test 中 `Missing days` 的占比高于全体语料，解释实验结果时应把这一随机构成差异视为潜在混杂变量。

## 数据检查、重复和标签覆盖

脚本逐句检查基本依存树。r2.18 的结果为：全部 1,004 句恰有一个 root；未发现 HEAD 指向无效整数 ID、非整数 HEAD、依存循环、缺失 `sent_id` 或重复 `sent_id`。没有发现 dev/test 中出现而 train 未出现的 DEPREL。完整句长、UPOS、DEPREL、来源分布和检查结果见 `data/processed/yue_hk/stats.json`。

共有 19 个重复 group，涉及 49 句；全部重复句均保留且同组不跨 split。其中 2 组标注不一致，未修改：

- `sent_id` 319 与 321（文本“等陣先，等陣先！”）：ID 3 标点的 HEAD 分别为 1 和 4。
- `sent_id` 412 与 453（文本“哎唷，哎吔，呢個冇人贊成嘅。”）：ID 1、3、5 的 HEAD 不同，ID 5 的 DEPREL 分别为 `nsubj` 和 `obj:periph`。

所有重复组、source key、逐单元格标注差异和 annotation-row 哈希均记录在 `stats.json`；没有静默修复或删除。

## 普通话平行语料泄漏风险

r2.18 的 Cantonese-HK 和 Chinese-HK 每句都带显式 `# parallel_id = hk/...`。脚本确认两边相关 ID 唯一后，仅按该官方字段相等建立映射，不使用行号或未经验证的 `sent_id` 一一对应假设。

`data/processed/yue_hk/mandarin_parallel_exclusions.json` 保存了全部 101 条 dev 和 100 条 test 的普通话对应句排除记录；本次 201 条全部成功解析，unresolved 为 0。普通话原始文件只用于产生这份证据清单，没有加入粤语 train。

重要限制：目前尚未检查未来普通话 checkpoint 的训练来源，因此**不能声称已排除预训练 parser 接触平行测试句的风险**。选择 checkpoint 时必须另行核查其训练 treebank、版本和数据谱系；排除清单只是为这一步提供可执行依据。

## 复现命令与覆盖策略

在项目根目录运行，无第三方 Python 依赖：

```bash
python3 scripts/prepare_yue_ud.py
```

若目标文件已存在且内容完全一致，脚本报告 `unchanged`；若任一现有目标内容不同，脚本会拒绝覆盖，要求通过 `--output-dir` 使用独立版本目录。因此不会静默覆盖已有的不同划分。

本次还在新的临时目录中用相同输入和 seed 完整重跑，并逐文件比较 SHA-256；六个生成文件全部一致：

| 文件 | SHA-256 |
|---|---|
| `train.conllu` | `b2d6b96af234f22825bb007a9a2a57ef49619c59747dc582dca7fdb1a4331a2d` |
| `dev.conllu` | `41bc28d903457e70a4747307e56b3eb7820f2e6d3ede10ab549fd4a85aef63b7` |
| `test.conllu` | `1d7b19ac4c0a75d58a80482413ff9cac63ed18917262c18fa1f92fc0f871c0a0` |
| `split_manifest.json` | `8012d471987e2913a3394ef9ac8cb0e62d0f456b9f15fe32fb2f0df9d8c65d06` |
| `stats.json` | `a23a65e76bba6b8e890eb8ed0a1ce2bd2cd4e58a25dc4f35c33c6096dbb4e91f` |
| `mandarin_parallel_exclusions.json` | `06f89da4f8589b5b9bb7a5fd4ac1c725ffc649e02a02e0902a110070f69b9de8` |

程序内验收同时验证：每个原始句子恰好出现一次；三个 split 的 source key 两两互斥；重复 group 不跨 split；每个输出句块与对应原始句块的 SHA-256 相同。结果写入 `split_manifest.json` 的 `validation` 字段。

## 产物

- `scripts/prepare_yue_ud.py`：无第三方依赖的数据准备脚本。
- `data/processed/yue_hk/train.conllu`
- `data/processed/yue_hk/dev.conllu`
- `data/processed/yue_hk/test.conllu`
- `data/processed/yue_hk/split_manifest.json`：版本、哈希、算法、每句 source key/位置/group/split、输出哈希和验收结果。
- `data/processed/yue_hk/stats.json`：split 统计、句长分布、UPOS/DEPREL、重复组、未见标签和数据问题。
- `data/processed/yue_hk/mandarin_parallel_exclusions.json`：dev/test 的普通话平行句排除清单及映射证据。

## Colab：全部 Stanza 中文 parser 的零样本 test LAS

`notebooks/evaluate_all_stanza_chinese_on_yue_test.ipynb` 可直接上传 Google Colab。建议选择 GPU runtime，然后按顺序运行全部单元格。Notebook 会自动从固定的官方 UD Cantonese-HK r2.18 URL 下载原始 1,004 句文件，核对原始 SHA-256，再按本项目 manifest 的固定原始位置重建当前 100 句 test 并核对 test SHA-256；无需手动上传数据，任一版本不符都会停止。

Notebook 固定 `stanza==1.14.0`，读取与该版本配套的官方 resources 清单，并去除 package 别名造成的重复后运行以下 4 个实际不同的中文 dependency parser 配置：

- 简体：`gsdsimp_charlm`
- 简体：`gsdsimp_nocharlm`
- 简体：`gsdsimp_electra-large`
- 繁体：`gsd_nocharlm`

评测使用粤语 gold 句界与 gold FORM 分词，但 POS、lemma、HEAD 和 DEPREL 均由相应 Stanza pipeline 预测。这样得到的是 parser 迁移的可比零样本基线，不是端到端中文 tokenizer 分数。主结果 `LAS_CoNLL18_percent` 来自固定 URL 和 SHA-256 的 CoNLL-2018 官方评测脚本；同时输出保留 DEPREL 子类型的严格 LAS、UAS、UPOS、每个预测 CoNLL-U、模型配置、输入与输出哈希。不能依赖 `from stanza.utils import conll18_ud_eval`，因为 Stanza 1.14.0 的 PyPI wheel 未导出该模块。

运行结束后会显示按 LAS 排序的比较表与柱状图，并由 Colab 下载 `yue_test_stanza_zh_results.zip`，其中包含 `results.csv`、`results.json`、比较图和各预测 CoNLL-U。`electra-large` 配置下载量与显存占用明显高于其余配置，Notebook 会逐个释放 pipeline 以降低峰值显存。Pipeline 使用 `DownloadMethod.REUSE_RESOURCES`：复用已经下载的 Stanza resources，同时允许首次获取 parser 所需的外部 Hugging Face Transformer；使用 `download_method=None` 会错误地令 FoundationCache 进入 `local_files_only` 模式。单个配置若因网络或显存失败会被写入 `failures` 并继续其余配置。这里没有训练或微调。

训练前已确认：该 notebook 固定 gold 句界/FORM 分词，但 POS 与 lemma 是 pipeline 预测值，depparse 消费预测 POS；评测对象是自定义 test，不是 dev。四个中文模型已在 test 上进行比较，因此该 test 已被用于模型族选择，不能再描述为从未查看。主指标预先固定为忽略 DEPREL subtype 的官方 CoNLL-2018 LAS，完整标签 strict LAS 为补充。详细的训练前约束、ELECTRA checkpoint 标识和后续 dev-only 选择规则见 `experiments/yue_dependency_transfer_protocol.md`。

## Colab：ELECTRA-large LoRA 粤语适配

`notebooks/finetune_yue_stanza_electra_lora.ipynb` 是独立、可直接上传 Colab 的首轮训练 notebook。它重建并校验本项目固定的 803/101/100 句 train/dev/test，先用固定的中文 ELECTRA POS 与 charlm lemma processor 产生一次 train/dev/test predicted-POS/lemma 缓存，随后不再把这两个 processor 放进训练过程。gold 句界、gold FORM 分词以及 gold HEAD/DEPREL 监督保持不变。

首轮配置预先固定为 LoRA `r=8, alpha=16, dropout=0.1`，Stanza 默认的四类 target module，parser AdamW 学习率 `1e-3`、LoRA 学习率 `2e-5`、最多 4,000 steps、每 100 steps 只看 dev 的官方 CoNLL-2018 LAS、600 steps 无提升停止。训练不做无标点复制增强。Notebook 会显式扩展普通话 checkpoint 缺少的 DEPREL 输出单元，只初始化新增单元；载入后逐张量断言旧 parsing 权重及旧标签切片完全相同，并断言 Transformer 基础参数冻结、只有 LoRA 参数可训练。

test 评分被隔离在最后一个单元格，只加载已经由 dev 选定的 checkpoint；运行后与已记录的 41.32% 作比较并下载完整 zip。41.32% 那次运行的 Hugging Face commit 尚未保存在本项目中，所以 notebook 会固定并记录当前运行解析到的完整 ELECTRA commit，但只有当它与旧运行记录相同时，才能声称与 41.32% 在完整 Transformer 版本上严格一致。更可靠的同版本效应量应在当前 notebook 固定的 revision 下另有预先记录的零步 baseline；本轮遵照约定不重复查看 test baseline。

`notebooks/yue_dev_diagnostics_only.ipynb` 是可直接 **Run all** 的独立诊断 notebook。它要求上传已有的 `yue_lora_electra_r8_results.zip`，只从归档提取 dev-best checkpoint；不读取自定义 train、不建立训练循环、不调用参数更新，也不评估 test。它只重建 dev，核对冻结 predicted-POS/lemma 缓存 SHA 后，分别在 predicted POS 与 gold POS/morph（lemma 仍为同一预测值）条件下进行推理，导出 POS 错误关联、依存距离、relation 混淆、最差句子和标注一致性人工核查候选。

## Colab：全量微调对照与 LoRA dev 比较

`notebooks/yue_full_finetune_vs_lora.ipynb` 从同一个原始普通话 Stanza ELECTRA-large dependency checkpoint 和固定 Hugging Face revision `d017e219578df8e4885484edbc8969dbdea9cbe0` 重新初始化。它只扩展 checkpoint 缺少的粤语 DEPREL 单元并逐张量验证旧 parsing 权重未变，然后更新完整 Transformer 与 parsing 层。POS/lemma processor 冻结，train/dev、predicted POS/lemma 缓存、seed、parser/Transformer 学习率、步数、early stopping 和 dev CoNLL-2018 LAS 选择规则均与 LoRA 运行匹配。

运行中需上传已经产生的 `yue_dev_diagnostics.zip`；notebook 会校验整个 zip 以及 LoRA dev prediction 的 SHA-256，只复用其中的 dev 预测，不会重新训练 LoRA。输出包含总体、距离和 relation 的 LoRA/full 对比，逐词 fixed/regressed 表，以及确定性抽取的 36 句人工核查材料。该 notebook 不计算 test；单次 matched-schedule、单 seed 结果只能判断这套日程下全量微调是否优于 LoRA，不能单独证明 LoRA 存在一般性的容量限制。建议使用 Colab A100，因为完整 ELECTRA-large 参数、梯度、优化器状态和最终 checkpoint 的资源需求远高于 LoRA。

## 普通话—粤语 dependency distribution（纯数据诊断）

`scripts/analyze_parallel_dependency_distributions.py` 不加载或修改模型。主分析只使用自定义粤语 train 的 803 句及其通过官方唯一 `parallel_id` 验证的 Chinese-HK 平行译文；dev/test 仅出现在明确标记为 descriptive 的全语料附录中，不用于选择模型设置。输出位于 `analysis/dependency_distribution_r2.18/`，包括 strict/base DEPREL、HEAD 方向、依存距离、条件 attachment signature 和逐平行句结构差异。需注意 Chinese-HK 并不是普通话 ELECTRA checkpoint 训练语料的替代证据，因此该分析描述的是两个平行 treebank 的 gold 标注分布，而非 checkpoint 内部表示。

## Wu 2018 普通话→粤语 Dependency-to-Dependency 复现/适配

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ss-sebastian/zh-yue-d2d/blob/main/notebooks/wu2018_mandarin_cantonese_replication.ipynb)

完整、自包含的 Colab notebook 位于 [`notebooks/wu2018_mandarin_cantonese_replication.ipynb`](notebooks/wu2018_mandarin_cantonese_replication.ipynb)。它自动下载并校验 UD r2.18 Chinese-HK/Cantonese-HK，仅按显式 `parallel_id` 恢复 1,004 个真实平行配对，实现 CES/HES、多流 GRU 编码器、Bahdanau attention、交互式 word/action GRU、DEV-only checkpoint 选择、四项 source-syntax 消融以及两种严格区分的评估模式。

### 低资源非投射扩展：保留全部 gold pairs

标准 Wu-style arc-standard 只能表示投射树，但 Cantonese-HK 中有 118/1,004（11.75%）棵非投射目标树；对如此小的数据集直接删除并不是无成本的清洗，而会造成明显的信息损失和选择偏差。本实验因此加入 Nivre (2009) 的 `SWAP` transition：它把 second-top stack item 移回 buffer，并用 gold tree 的 projective order 构造确定性 oracle。被 SWAP 后重新 SHIFT 的 token 复用已有 Word-RNN representation，不会被翻译器重复生成；只有首次 SHIFT 才触发粤语 word generation。

这是**相对于 Wu et al. 基线、面向本低资源平行树库的实验创新/必要适配**，并不是声称 `SWAP` 操作本身由本项目首创。全语料单元测试确认 1,004/1,004 棵粤语树均可通过 `gold tree → actions（含 SWAP）→ reconstructed tree` 精确恢复 HEAD 与 DEPREL，因此训练、DEV 和 TEST 不再排除任何非投射 pair。理论依据见 Joakim Nivre (2009), [Non-Projective Dependency Parsing in Expected Linear Time](https://aclanthology.org/P09-1040/)。
