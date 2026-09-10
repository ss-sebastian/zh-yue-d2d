# 粤语 dependency transfer：训练前固定协议

记录日期：2026-09-08（Asia/Hong_Kong）

## 已确认的零样本 baseline 条件

当前 Colab 表不是 gold-POS dependency-only baseline，而是 **gold 句界 + gold 分词、预测 POS/lemma 的 pipeline-style baseline**。

- split：自定义 `test`，不是 `dev`。
- test 文件：`data/processed/yue_hk/test.conllu`。
- test SHA-256：`1d7b19ac4c0a75d58a80482413ff9cac63ed18917262c18fa1f92fc0f871c0a0`。
- 规模：100 句、1,360 个整数 ID 词节点。
- 句界与分词：固定为 gold。Notebook 从每句整数 ID 行抽取 FORM，形成 `list[list[str]]`，并以 `tokenize_pretokenized=True` 调用 Stanza；写出预测前逐句检查词数和 FORM 完全相同，不同即终止。
- POS：预测。pipeline 顺序为 `tokenize,pos,lemma,depparse`，depparse 消费对应中文 POS processor 的预测 UPOS/XPOS/FEATS。gold UPOS 没有送入 pipeline，只用于评分和“不含标点”筛选。
- lemma：由对应 Stanza lemma processor 预测。
- dependency：Stanza depparse 预测 HEAD 与 DEPREL。

因此表中的 UPOS 63–77% 与实现一致。它证明 POS 被预测，但不是固定分词的证据；固定分词由 pretokenized 输入和 FORM/词数断言独立保证。

如果以后要回答纯 dependency 层迁移问题，可另建 gold 分词 + gold POS baseline；本阶段不扩展该实验。LoRA 前后主比较必须保持本节完全相同的 predicted-POS pipeline 条件，不能把当前 baseline 与 gold-POS 微调结果比较。

## 预先指定的指标

主指标：**CoNLL-2018 LAS（%）**，包含所有对齐词节点，包括标点。固定 evaluator：

- URL：`https://universaldependencies.org/conll18/conll18_ud_eval.py`
- SHA-256：`1072e02af00b1a56205b5e8216d51dee9b8944a104d80744afaccc78859fcb16`

该 evaluator 在载入 gold 和 system 时均执行 `DEPREL.split(":")[0]`，因此主指标忽略冒号后的语言特定 relation subtype。

补充指标：

1. 完整 DEPREL strict LAS：HEAD 与完整 DEPREL 字符串都相同才算正确，包含标点。
2. 完整 DEPREL strict LAS（不含标点）：同上，但排除 gold UPOS 为 `PUNCT` 的节点。

ELECTRA-large 的本次 Colab test 结果（用户报告的表中数值）为：

| 指标 | 分数 |
|---|---:|
| CoNLL-2018 LAS（主） | 41.32% |
| 完整 DEPREL strict LAS | 37.94% |
| 完整 DEPREL strict LAS，不含标点 | 36.11% |

这些差值与 subtype 是否匹配、标点依存是否正确均有关，不能仅凭三个总体分数把全部差值归因于某一种错误；需要逐标签误差分析才能分解原因。本阶段不扩展该分析。

## test 使用记录与模型选择边界

这张表来自 test，并且四个中文模型配置已在 test 上比较，ELECTRA-large 的选择也参考了该比较。因此：

- 不得再把该 test 描述成“完全未用于模型选择”或“从未查看”。
- 41.32% 可作为已记录的迁移前 test baseline，但不是在完全未触碰 test 条件下得到的无偏模型选择后估计。
- 从此记录之后，LoRA 超参数、epoch、early stopping、随机种子决策与 checkpoint 选择只能根据 dev。
- test 不应被反复查看；训练协议冻结后只做预先约定的最终评测。
- 这项历史使用无法通过后续“不看 test”撤销。若论文需要严格独立、从未用于模型族选择的最终测试，必须另有未触碰的外部评测集；当前项目没有这样的集合。

当前结果不能用于判断是否超过 Franklin，也不能据此预测 LoRA 的提升幅度。

## 固定的 ELECTRA 起点

后续粤语适配使用 Stanza 1.14.0 `zh-hans` 的同一套 `default_accurate`/ELECTRA 配置：

- tokenizer：`gsdsimp`，官方 registry MD5 `48f993223d568afedc2893f7cd76719c`。
- POS：`gsdsimp_electra-large`，官方 registry MD5 `73859e5ec15bedc545d6deafd6ddba94`。
- lemma：`gsdsimp_charlm`，官方 registry MD5 `b49edd41abb063a87b125ec53aa5b96c`。
- dependency parser：`gsdsimp_electra-large`，官方 registry MD5 `c7ea98d93459b22720337a9dcba1a848`。
- Transformer：`hfl/chinese-electra-180g-large-discriminator`，由 Stanza 1.14.0 的 `default_packages.py` 指定。
- Stanza resources 1.14.0 JSON SHA-256：`4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c`。

上述 MD5 固定 Stanza 分发的 processor 文件，但 Hugging Face 模型名本身仍是浮动引用。更新后的 Colab 会在每项结果的 `processor_artifacts` 中保存实际 `.pt` 文件 SHA-256，并在 `transformer_cached_revisions` 中保存本次缓存的 Hugging Face commit revision。正式训练前必须确认 ELECTRA 只对应一个 cached revision，并在适配脚本中固定该 revision；若列表为空或包含多个 revision，应先消除歧义。在这一步完成前，不应声称 Transformer 权重已被完整固定。

## 首轮适配实现

训练入口为 `notebooks/finetune_yue_stanza_electra_lora.ipynb`。固定配置：seed 42；LoRA `r=8, alpha=16, dropout=0.1`；target modules 为 `query,value,output.dense,intermediate.dense`；parser AdamW 学习率 `1e-3`；LoRA 学习率 `2e-5`；最多 4,000 steps；每 100 steps 用 dev 官方 CoNLL-2018 LAS 评估；600 steps 无提升停止；同分时保留较早 checkpoint；不做 `augment_nopunct`。

Notebook 先用固定 processor 生成 predicted POS/lemma CoNLL-U 缓存，缓存仍保留 gold HEAD/DEPREL 作为监督。缓存完成后释放 POS/lemma pipeline；训练仅载入 dependency parser。普通话 DEPREL 词表不存在的粤语 train 标签被确定性追加，旧标签编号不变；仅新增输出切片初始化为零。载入兼容 checkpoint 后代码逐张量检查原 parsing 参数和旧标签输出切片完全相等，并检查 Transformer 基础参数冻结、LoRA 与 parser 参数可训练。

checkpoint 选择只读 dev。test dependency 评估只在 notebook 最后单元格进行一次，并继续披露 test 已用于模型族选择。不得用该结果回头改变本轮配置。当前项目没有保存产生 41.32% 的 Hugging Face ELECTRA 完整 commit；新 notebook 会隔离缓存、解析一次并保存完整 commit SHA，随后强制离线复用。除非旧结果记录中的 commit 与它相同，否则“相对 41.32% 的变化”还混有未解决的 Transformer 版本因素，不能完全归因于 LoRA。
