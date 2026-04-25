# Engineering Specification

Source PDF: `C:\Users\ASVS\Desktop\1\kan_engineering_original_RA_NUS.pdf`

跨模态哈希项目工程化实现文档
最终定稿（聊天验收版整理成文）
面向 Codex 的唯一正式工程规范
文档状态：最终定稿
适用范围：从创建文件夹后的下一步开始，覆盖 Stage 0 到 Stage 7，全链路适用于 MIRFlickr-25K、
NUS-WIDE、MSCOCO。
最高目标：在完全以当前已完成成果为基础的前提下，最大化三个数据集最终训练的 mAP，并保证
Codex 在实现过程中不存在歧义、不会走偏、不会制造无用代码或污染仓库。
本文件是当前项目唯一正式工程规范。历史主文档、Stage 2 补充、诊断附录以及后续定稿的方法/损失
文稿，均已整合进本文件；若后续与旧文档冲突，一律以本文件为准。

目录

## 0. 文档身份、适用范围与最高目标
1. 规范优先级与全局执行原则
2. 正式符号系统与 Python 标识符
3. 仓库结构、源码/产物边界与 Git 卫生

## 4. Stage 0：启动、环境、CLIP 权重与原始数据准备

## 5. Stage 1：正式预处理、统一样本构造与 split 冻结

## 6. Stage 2：正式特征提取（吸收 baseline 产物）

## 7. Stage 3：正式语义相关性模块

## 8. Stage 4：正式模型主链

## 9. Stage 5：正式损失协议

## 10. Stage 6：正式训练协议

## 11. Stage 7：正式评估协议与目标门槛

## 12. 三数据集正式 profile（第一版默认配置）

## 13. 更新后的阶段化诊断协议

## 14. Codex 执行协议（强约束版）

## 15. 最终总验收清单

## 附录 A：数据来源与原始目录摘要

## 附录 B：Codex 单轮任务模板

## 附录 C：正式产物总表

## 0. 文档身份、适用范围与最高目标
本文件用于指导当前跨模态哈希项目的完整工程落地，服务对象包括人工审阅者、Codex 执行者、后续
训练与评估调用者。它不是论文正文，而是正式工程规范。凡属于论文或你后续定稿文档中已经写清楚
的内容，视为硬约束；凡属于工程必须落地但原文未写清的部分，必须在本文件中显式补齐，禁止隐藏
在代码实现里。
本文件覆盖三个层面：
- 第一层：从 Stage 0 到 Stage 7 的阶段输入、输出、门禁与 validator。
- 第二层：正式符号、代码命名、配置命名、目录职责、Git 卫生。

- 第三层：三数据集差异化 profile、阶段化诊断协议、Codex 的执行约束。
本项目的最高目标不是“先把代码堆起来”，而是：以你当前已经做出来的方法成果为基础，在保证实
现严格、清晰、可审计的前提下，尽可能让 MIRFlickr-25K、NUS-WIDE、MSCOCO 三个数据集的最终
mAP 接近或超过 RA 论文结果。
1. 规范优先级与全局执行原则

### 1.1 规范优先级
1. 你在当前对话中最新明确确认的要求。
2. 本最终定稿工程化实现文档。
3. 你最新定稿的方法文档与损失文档。
4. 旧工程化文档、Stage 2 补充协议、旧诊断附录。
5. RA 论文（仅数据集、参考结果与实验口径参考，不再支配 Stage 3 之后的模型设计）。

### 1.2 全局执行原则
- 阶段门禁原则：上一阶段未通过 validator，不允许进入下一阶段。
- 唯一对象原则：Stage 1 冻结样本集合与顺序；Stage 2 冻结特征；Stage 3 冻结正式监督矩阵 S；
后续阶段只能消费冻结产物，不得回写上游。
- fail-fast 原则：缺文件、缺字段、数量不闭合、路径错误、样本顺序漂移、出现 NaN/Inf、模型权重
不可用，都必须立即报错停止。
- 禁止 silent fallback：禁止自动跳过坏样本、自动补零向量、自动改模型、自动换 split、自动把
CPU 结果当正式结果。
- 代码简洁原则：禁止为未来可能扩展预铺大量空壳类、空 helper、双份实现、兼容层、历史别名
层。
- 统一 split 原则：正式 split 只承认 train、query、retrieval。
- 最高目标原则：允许三数据集采用不同 profile，但主链框架、正式符号、评测口径、Git 规则与诊
断协议必须统一。
2. 正式符号系统与 Python 标识符

### 2.1 正式数学符号
- 输入与特征：XI, XT, ZI, ZT, YI, YT, FI, FT, GI, GT, HI, HT, BI, BT。
- 语义监督：A, R, Se, C, S，其中 S = C ⊙ Se。

- 关系预测与同模态诱导目标：P_IT, P_II, P_TT, S_II_star, S_TT_star。
- 损失：L_IT, L_II, L_TT, L_sem, L_pair, L_q, L_bal, L_total。

### 2.2 Python 标识符
- X_I, X_T, Z_I, Z_T, Y_I, Y_T, F_I, F_T, G_I, G_T, H_I, H_T, B_I, B_T。
- A, R, Se, C, S。
- P_IT, P_II, P_TT, S_II_star, S_TT_star。
- L_IT, L_II, L_TT, L_sem, L_pair, L_q, L_bal, L_total。

### 2.3 废弃与禁止写法
- 禁止使用 S_tilde、tilde_S、S_hat 作为主符号。
- 禁止继续把历史 Psi 作为主预测矩阵名；当前正式统一为 P_IT。
- 禁止使用 P_TI。
- 禁止使用 x_I、x_T 这种既可能表示行向量又可能表示整体矩阵的模糊命名。
- 禁止通过 compatibility layer 同时保留新旧别名。

### 2.4 配置项命名
- lambda_ar_fusion：Se = λA + (1-λ)R 中的 λ。
- tau_confidence：双向可信度中的 softmax 温度 τ。
- alpha_intra_topology：L_sem 中同模态拓扑项权重 α。
- beta_relation_weight：加权关系损失中 W = 1 + β target 的 β。
- lambda_sem_total、lambda_pair_total、lambda_q_total、lambda_bal_total：总损失四个外
层权重。
正式要求：数学上的 λ 不允许在代码里复用为总损失权重名。
3. 仓库结构、源码/产物边界与 Git 卫生

### 3.1 正式目录
configs/
datasets/
stages/
profiles/
docs/
scripts/
src/
datasets/
builders/
validators/
features/
semantic/

models/
encoders/
tree/
graph/
heads/
wrappers/
losses/
trainers/
eval/
utils/
data/
raw/
mirflickr25k/
nuswide/
mscoco/
processed/
mirflickr25k/
nuswide/
mscoco/
outputs/

### 3.2 源码与产物的硬边界
源码目录只能包含可复用逻辑；产物目录只能包含运行输出。禁止把以下内容提交到源码目录：
- *.npy、*.npz、*.pt、*.pth、*.ckpt。
- wandb、tensorboard、logs。
- reports 中非正式模板文件。
- 临时下载包、临时解压目录。
- 一次性 debug dump。

### 3.3 .gitignore 正式要求
- data/raw 中的大型下载包与解压原始二进制。
- data/processed 整体。
- outputs 整体。
- *.pt、*.pth、*.ckpt、*.npy、*.npz、*.pkl、*.log。
- .ipynb_checkpoints、__pycache__。

### 3.4 垃圾内容的正式定义
1. 未被当前正式 stage runner 调用的临时脚本。
2. 未被任何正式配置引用的重复 config。
3. 被新实现替代但仍保留的旧 alias 模块。
4. 调试专用 notebook、scratch 文件。
5. 运行产物误落到源码目录。

6. 注释掉的大段旧逻辑。
7. 无入口引用的 helper。
8. 临时 print、临时路径硬编码、临时 patch 分支。
9. 以 backup、copy、final2、newnew 命名的重复文件。
正式例外：协议文档、正式 config、正式 validator、正式 summary 模板、当前 best checkpoint、当
前 formal run 必需 cache，不算垃圾内容。

## 4. Stage 0：启动、环境、CLIP 权重与原始数据准备
Stage 0 从创建文件夹后的下一步开始，且必须在 Stage 1 之前完成。

### 4.1 Stage 0 目标
1. 固定环境与依赖记录。
2. 准备 CLIP 权重。
3. 下载或放置三数据集原始数据。
4. 通过 raw validator。

### 4.2 环境冻结协议
正式运行前必须产出 docs/environment_lock.md 与 outputs/env/environment.lock.json。最少记录
Python、torch、torchvision、transformers、CUDA、faiss、numpy、pillow、scipy 版本，以及
GPU 型号、显存和操作系统信息。
正式默认：Stage 2 到 Stage 7 使用 float32；正式运行默认不开 AMP；AMP 只允许在 float32 formal
run 已通过后，作为显式实验分支启用。

### 4.3 CLIP 权重准备协议
正式 backbone 固定为 openai/clip-vit-base-patch32。若 model_local_path 存在，则必须优先本地
加载；若本地不存在，允许从正式模型标识下载；若环境禁止联网且无本地权重，Stage 0 直接失败。
禁止自动切换为 OpenCLIP、SigLIP、ResNet 或其他 CLIP 变体。

### 4.4 三数据集下载与原始目录协议

#### 4.4.1 MIRFlickr-25K
官方 MIRFLICKR 下载页提供 mirflickr25k.zip（image collection，含 tags 与 EXIF）和
mirflickr25k_annotations_v080.zip（annotations）。
data/raw/mirflickr25k/

mirflickr25k.zip
mirflickr25k_annotations_v080.zip
extracted/
images/
meta/
tags/
exif/
annotations/
README.txt

#### 4.4.2 MSCOCO
当前工程口径使用 2014 train + val 图像级 pair，因此原始输入固定为 train2014.zip、val2014.zip、
annotations_trainval2014.zip。
data/raw/mscoco/
train2014.zip
val2014.zip
annotations_trainval2014.zip
extracted/
train2014/
val2014/
annotations/
captions_train2014.json
captions_val2014.json
instances_train2014.json
instances_val2014.json

#### 4.4.3 NUS-WIDE
NUS-WIDE 官方当前说明重点覆盖标签与 tag 矩阵；工程上必须把图像二进制本地镜像视为单独
prerequisite。
data/raw/nuswide/
NUS-WIDE.zip
NUS_WID_Tags.zip
extracted/
Groundtruth/
ConceptsList/
tags/
Final_Tag_List.txt
All_Tags.txt
AllTags81.txt
AllTags1k.txt
TagList1k.txt
images/
image_index.tsv
image_index.tsv 为强制文件，字段为 raw_index<TAB>image_relative_path，它是把官方顺序与本
地文件系统绑定的唯一正式映射文件。若 images 或 image_index.tsv 缺失，Stage 0 必须硬失败。

### 4.5 Stage 0 raw validator
每个数据集都必须产出 raw_audit.json 和 raw_validator_summary.json。必须检查：必需压缩包是
否存在、必需解压目录是否存在、关键 json/txt 是否存在、关键文件行数/条数是否符合协议、NUS

image_index.tsv 行数是否为 269648、NUS All_Tags.txt 行数是否为 269648、COCO
train2014+val2014 能否闭合到 123287 图像级 pair、MIR 原始图像总数是否为 25000。Stage 0 未通
过，不得进入 Stage 1。

## 5. Stage 1：正式预处理、统一样本构造与 split 冻结
当前工程文档中关于三数据集过滤与 split 的主规则继续保留；本节新增 raw 到统一样本的细节规范，
使其可直接编码。

### 5.1 Stage 1 唯一输入
只允许读取 data/raw/<dataset> 下的正式 raw 输入与对应 dataset config。

### 5.2 统一样本字段
- sample_id
- dataset_name
- image_path
- text_source
- label_vector
- raw_index
- meta

### 5.3 sample_id 正式规则
- MIR：mir_{raw_index:05d}
- NUS：nus_{raw_index:06d}
- COCO：coco_{image_id:012d}
禁止使用随机 UUID。禁止 Stage 2 及之后重写 sample_id。

### 5.4 三数据集正式构造规则

#### 5.4.1 MIRFlickr-25K
- 一张图 + tags 文本 + 24 维标签 = 一条样本。
- text_source：按原顺序取 tags，去空白与空 token，用单个空格连接。
- 空文本样本不能进入 manifest_filtered.jsonl。
- label_vector 为 24 维二值，同时写入 meta.annotation_positive_count。

正式过滤规则继续使用 pragmatic_high_signal_v1：先删空文本，再计算 raw_tag_token_count 与
annotation_positive_count，按 raw_tag_token_count desc、annotation_positive_count desc、
sample_id asc 稳定排序，截断到 20015 条。

#### 5.4.2 NUS-WIDE
NUS-WIDE 保留当前 top-10 概念子集策略，但修复文本定义歧义。
- label_vector_raw：由 81 个 concept label 文件拼接。
- concept_subset：按正样本数降序，若并列按类别名字典序，取前 10 个类别。
- 正式 label_vector：只保留这 10 维。
- 仅保留 sum(label_vector) > 0 的样本。
- 过滤后数量必须严格等于 186577。
NUS 文本恢复正式协议：读取 Final_Tag_List.txt 得到 5018 个 tag 词表；读取 All_Tags.txt 第
raw_index 行二值向量；取所有值为 1 的 tag；按词表原顺序恢复，并用单个空格连接成
text_source。若某行全 0，则写入 raw_empty_tag_row = true，不得 silently 补 token。正式要求：
NUS 的 text_source 只能来自“二值矩阵 + 词表反解”，绝不能把 All_Tags.txt 直接当自然语言逐行读
取。

#### 5.4.3 MSCOCO
- 正式采用图像级 pair。
- 同一图像全部 captions 按 annotation id 升序排列后，用“. ”连接成单条 text_source。
- label_vector 为 80 维 category multi-hot。
- 总样本数必须严格等于 123287。
- 工程上正式解释为 train ⊂ retrieval，以保证数量闭合。
COCO category multi-hot 只允许从 instances_train2014.json 与 instances_val2014.json 构造；
caption 只允许从 captions_train2014.json 与 captions_val2014.json 构造。

### 5.5 split 正式规则
1. 对最终样本按 sample_id 升序排序。
2. 使用 seed = 0 做 permutation。
3. 前 2000 为 query。
4. 剩余为 retrieval。
5. retrieval 前 5000 为 train。
适用于 MIR（20015→2000/18015/5000）、NUS（186577→2000/184577/5000）和 COCO
（123287→2000/121287/5000）。

### 5.6 Stage 1 输出文件
data/processed/<dataset>/manifest/
manifest_raw.jsonl
manifest_filtered.jsonl
manifest_meta.json
data/processed/<dataset>/splits/
query_ids.txt
retrieval_ids.txt
train_ids.txt
split_summary.json
data/processed/<dataset>/reports/
preprocess_summary.json
validator_summary.json
config_snapshot.json
order_hashes.json

### 5.7 Stage 1 validator 额外必须检查
- sample_id 格式是否符合正式命名规则。
- NUS text_source 是否由 5018-tag 反解得到。
- NUS concept_subset 名称与顺序是否冻结。
- COCO 每个样本是否至少有 1 条 caption。
- 三数据集的 sample_id_order_sha256 是否写入 order_hashes.json。

## 6. Stage 2：正式特征提取（吸收 baseline 产物）
Stage 2 以历史正式补充协议为基础，只做两件事：保留全部硬约束；把 baseline 与 gap 诊断产物纳入
正式交付。

### 6.1 唯一输入边界
只允许读取 manifest_filtered.jsonl 与 query_ids.txt、retrieval_ids.txt、train_ids.txt。严禁重排样
本、重写 text_source、重写 label_vector、绕过 Stage 1 直接从 raw 造样本。

### 6.2 正式 backbone 与预处理协议
- backbone 固定为 openai/clip-vit-base-patch32。
- 图像：RGB -> shortest side resize 224 -> bicubic -> center crop 224x224 -> CLIP mean/std
normalize。
- 文本：tokenizer = openai/clip-vit-base-patch32，max_length=77，padding=max_length，
truncation=true，return_attention_mask=true。
- 图像使用 model.get_image_features，文本使用 model.get_text_features。
- 取得 projected embedding 后，每一行必须执行 L2 normalize。

- 输出维度固定为 512，dtype 固定为 float32，正式 device 固定为 cuda:0。

### 6.3 正式输出
data/processed/<dataset>/feature_cache/clip_vit_b32_formal_v1/
X_I.npy
X_T.npy
meta.json
validator_summary.json
baseline_summary.json

### 6.4 baseline_summary.json 必须字段
- dataset, feature_set_id, filtered_count, query_count, retrieval_count。
- paired_cosine_mean, paired_cosine_median。
- random_cosine_mean, random_cosine_median。
- cosine_gap_mean, cosine_gap_median。
- clip_i2t_map_at_50, clip_t2i_map_at_50。
- block_size_similarity, baseline_completed, failure_reason。

### 6.5 Stage 2 baseline 计算正式协议
由于 COCO 与 NUS retrieval 很大，正式 baseline 计算不允许一次性 materialize 全部 query ×
retrieval 相似矩阵。必须采用 blockwise exact cosine retrieval：query 逐块、retrieval 逐块，分别得
到 I→T 与 T→I 的 raw CLIP mAP@50。

### 6.6 Stage 2 stop/go
- X_I/X_T 形状正确。
- dtype 正确。
- 无 NaN/Inf。
- 每行 L2 norm 约等于 1。
- 顺序与 manifest 一致。
- paired_cosine_mean > random_cosine_mean。
- paired_cosine_median > random_cosine_median。
若 gap 不为正，Stage 2 直接判失败。

## 7. Stage 3：正式语义相关性模块
Stage 3 以后，正式数学主线完全以你后续定稿文档为准，不再沿用旧工程文档里的 S_tilde 主线。

### 7.1 Stage 3 唯一输入边界
Stage 3 只允许读取：Stage 2 的 X_I.npy / X_T.npy、Stage 1 的 train_ids.txt、Stage 1 的
manifest_filtered.jsonl。正式训练构造域只允许是 train split（5000 对）。严禁在 filtered 全集上构
造正式训练监督矩阵，严禁把 query/retrieval 混入 Stage 3 正式监督。

### 7.2 正式输出文件
data/processed/<dataset>/semantic_cache/se_c_s_formal_v1/
A.npy
R.npy
Se.npy
C.npy
S.npy
meta.json
validator_summary.json
semantic_diagnostics.json
Omega_topk_diag.npz

### 7.3 正式数学定义
A_ij = (1 + <x_i^I, x_j^T>) / 2
M_I = X_I X_I^T
M_T = X_T X_T^T
R_ij = 0.5 * (1 + ((M_I)_{i,:} · (M_T)_{j,:}) /
(||(M_I)_{i,:}||_2 * ||(M_T)_{j,:}||_2)) )
Se = lambda_ar_fusion * A + (1 - lambda_ar_fusion) * R
P^{I->T}_{ij} = exp(Se_ij / tau_confidence) / sum_k exp(Se_ik / tau_confidence)
P^{T->I}_{ij} = exp(Se_ij / tau_confidence) / sum_k exp(Se_kj / tau_confidence)
C_ij = sqrt(P^{I->T}_{ij} * P^{T->I}_{ij})
S = C ⊙ Se

### 7.4 Stage 3 数据类型与值域
- A, R, Se, C, S 全部保存为 float32。
- 形状统一为 [5000, 5000]。
- 期望值域：A, R, Se, S 在 [0,1]；C 在 (0,1]。

### 7.5 semantic_diagnostics.json 必须字段
- train_count, shape_ok。
- range_a_ok, range_r_ok, range_se_ok, range_c_ok, range_s_ok。
- diag_mean_s, offdiag_mean_s。
- paired_diag_quantile_in_row, paired_diag_quantile_in_col。
- row_topk_coverage, col_topk_coverage。

- topk_for_diagnostics, semantic_validator_passed, failure_reason。

### 7.6 Omega_topk_diag.npz 的地位
对每行取 S 的 top-k、对每列取 S 的 top-k，再与全部对角边并集，得到 Omega_topk_diag。它只用
于诊断 paired edge、row/col coverage 与 NUS 单向 coverage 问题；它不是正式训练监督本体。正式
训练监督对象永远是 S。

### 7.7 Stage 3 stop/go
- A/R/Se/C/S 形状正确。
- 值域正确。
- 无 NaN/Inf。
- diag_mean_s > offdiag_mean_s。
- row_topk_coverage > 0。
- col_topk_coverage > 0。
- paired diagonal 在行/列中的分位数不能整体崩塌。
若 Stage 3 不健康，必须先修 Stage 3，不允许先调后续损失。

## 8. Stage 4：正式模型主链（ChebyKAN -> 递归语义树 -> 图结构细化
-> 哈希码）
Stage 4 以后，不再使用 RA 原始 FKAN/BNE 作为你的正式模型主干；正式来源是你定稿的“语义模块
之后的完整新方案”。

### 8.1 Stage 4 唯一输入
只允许读取 Stage 2 的 X_I_train / X_T_train、Stage 3 的 S、Stage 3 的 meta / diagnostics，以及
dataset profile。严禁重新读 raw 图像、raw 文本、raw 标签进入训练主链。

### 8.2 Stage 4 正式模块文件边界
src/models/encoders/chebykan.py
src/models/tree/recursive_semantic_tree.py
src/models/graph/knn_graph.py
src/models/graph/graph_refiner.py
src/models/heads/hash_head.py
src/models/wrappers/cross_modal_hash_net.py

### 8.3 ChebyKAN 正式定义
对任一模态 M∈{I,T}，输入 X_M∈R^{N×512}。先执行 clamp(-1,1) 保证分量有界，再做 Chebyshev
多项式展开，得到 Z_M∈R^{N×d_z}。由于 Stage 2 CLIP 特征已 L2 normalize，正式实现不允许再做
dataset-statistics 重缩放。

### 8.4 递归语义树正式定义
- 必须包含第 l 层原型 P^(l)。
- 必须包含软分配 Π_I^(l)、Π_T^(l)。
- 必须包含同层共享直接语义节点 Ubar^(l)。
- 必须包含树递推 U^(l)。
- 必须包含自顶向下回写 Y_I、Y_T。
禁止把树层与图层揉成单一黑盒模块；树负责层级语义，图负责局部几何，这条边界必须保留。

### 8.5 图结构细化正式定义
F_I = LayerNorm(Z_I + beta_tree_injection * Y_I)
F_T = LayerNorm(Z_T + beta_tree_injection * Y_T)
[g_M]_{ij} = 1(j in N_k(i) or i in N_k(j)) * (1 + cos(f_{M,i}, f_{M,j})) / 2
N_M = D_tilde_M^{-1/2} * G_tilde_M * D_tilde_M^{-1/2}
H_I = tanh(N_I * F_I * W_I^(g) * W_I^(h))
H_T = tanh(N_T * F_T * W_T^(g) * W_T^(h))
B_I = sign(H_I)
B_T = sign(H_T)

### 8.6 训练态与推理态图构建协议

#### 8.6.1 训练态
正式训练模式固定为 train-split full-batch mode。原因：train split 恒为 5000 对，S 为 5000×5000，
full-batch 能完整保留 P_IT/P_II/P_TT 与 S/S_II_star/S_TT_star 的数学对应关系，并彻底消灭 batch
gather 错位。正式模式不使用 mini-batch 训练。

#### 8.6.2 推理态（query / retrieval）
推理阶段不再使用全量稠密图。query split（2000）允许 exact kNN；retrieval split 在 MIR 为
18015、COCO 为 121287、NUS 为 184577，其中 COCO/NUS 禁止 O(N^2) 全量稠密图。
- 当 split 大小 N <= 20000 时，允许 exact kNN。
- 当 N > 20000 时，必须使用 FAISS 或等价 ANN 工具构建 approximate top-k 邻域。
- 检索图必须是 split-local 图：retrieval 图只由 retrieval 内部样本构造，query 图只由 query 内部样
本构造。

- 不允许在 formal mode 里把 retrieval 节点混入 query 图编码过程。

### 8.7 Stage 4 禁止项
- 禁止把 Stage 2 CLIP backbone 拉回训练图中。
- 禁止把树和图合并成一个不可解释 block。
- 禁止训练时重新生成或更新 S。
- 禁止 mini-batch gather 作为 formal mode 默认实现。
- 禁止把推理时的大规模 kNN 写成全量 dense cosine。

## 9. Stage 5：正式损失协议
当前损失以你定稿文稿为唯一来源。

### 9.1 Stage 5 核心原则
- 跨模态监督只直接监督跨模态对象。
- 同模态监督必须由 S 诱导，不允许直接拿 S 去监督 P_II / P_TT。
- 正式模式使用 full-batch，因此 S、S_II_star、S_TT_star、P_IT、P_II、P_TT 全部在 5000×5000
全局 train 空间上定义。

### 9.2 关系预测矩阵
hat_h_i^I = h_i^I / (||h_i^I||_2 + eps)
hat_h_i^T = h_i^T / (||h_i^T||_2 + eps)
P_IT = (1 + hat_H_I * hat_H_T^T) / 2
P_II = (1 + hat_H_I * hat_H_I^T) / 2
P_TT = (1 + hat_H_T * hat_H_T^T) / 2

### 9.3 从 S 诱导同模态目标
Q_I = row_normalize(S)
Q_T = row_normalize(S^T)
S_II_star = Q_I * Q_I^T
S_TT_star = Q_T * Q_T^T
S_II_star 与 S_TT_star 必须在 trainer 初始化时由 full S 计算一次，并保存到
outputs/<run_id>/derived_supervision/。

### 9.4 加权关系损失
W_IT = 1 + beta_relation_weight * S
W_II = 1 + beta_relation_weight * S_II_star
W_TT = 1 + beta_relation_weight * S_TT_star
L_IT = ||sqrt(W_IT) ⊙ (P_IT - S)||_F^2 / ||W_IT||_1
L_II = ||sqrt(W_II) ⊙ (P_II - S_II_star)||_F^2 / ||W_II||_1

L_TT = ||sqrt(W_TT) ⊙ (P_TT - S_TT_star)||_F^2 / ||W_TT||_1
L_sem = L_IT + alpha_intra_topology / 2 * (L_II + L_TT)

### 9.5 配对一致性损失
L_pair = ||hat_H_I - hat_H_T||_F^2 / N

### 9.6 量化损失
L_q = ( ||H_I ⊙ H_I - 1||_F^2 + ||H_T ⊙ H_T - 1||_F^2 ) / (N * K)

### 9.7 比特平衡损失
L_bal = ( || (1/N) * 1^T H_I ||_2^2 + || (1/N) * 1^T H_T ||_2^2 ) / K

### 9.8 总损失
L_total =
lambda_sem_total  * L_sem  +
lambda_pair_total * L_pair +
lambda_q_total    * L_q    +
lambda_bal_total  * L_bal

## 10. Stage 6：正式训练协议

### 10.1 正式训练模式
- formal mode 固定为 train split full-batch。
- float32。
- AdamW。
- cosine scheduler with warmup。
- no AMP。
- no gradient accumulation。
- no mini-batch dataloader for train loop。

### 10.2 推荐优化器协议
- optimizer = AdamW。
- gradient clip = 5.0。
- weight decay 由 dataset profile 指定。
- warmup epochs = 5。
- scheduler = cosine decay。

### 10.3 训练脚本边界
scripts/train_formal.py

该脚本只读取 Stage 2 formal feature cache、Stage 3 formal semantic cache、dataset profile 与
run config；不得再读取 raw 数据。

### 10.4 检查点协议
outputs/<run_id>/
config_snapshot.json
train_log.jsonl
epoch_metrics.json
best_checkpoint.pt
last_checkpoint.pt
derived_supervision/
eval_reports/

### 10.5 run mode
- run_mode = dev：允许基于 query/retrieval 结果挑 profile。
- run_mode = formal_report：profile 冻结后重跑，不允许边跑边改 profile。
最终对外报告只能引用 formal_report。

## 11. Stage 7：正式评估协议与目标门槛

### 11.1 正式评估对象
- query 图像 -> retrieval 文本（I->T）。
- query 文本 -> retrieval 图像（T->I）。

### 11.2 正式编码与距离
- 正式存盘码：B_I, B_T，int8，元素为 {-1,+1}。
- 正式检索排名：Hamming distance primary。
- 允许内部为提速打包成 uint64，但正式缓存的人类可读版本仍为 int8 {-1,+1}。
- H_I/H_T 的 cosine 排名只允许做诊断，不得作为最终主报告。

### 11.3 正式评估指标
- mAP@50。
- Top-R precision。
- I->T 和 T->I 两个方向都要有。
- bit 长度：16 / 32 / 64 / 128。

### 11.4 正式评估输出
outputs/<run_id>/eval_reports/
metrics_summary.json
metrics_by_bits.json

top_r_curves.json
retrieval_examples/

### 11.5 RA 论文参考目标表（必须写入配置）
下表来自 RA 论文 Table 1，作为 reference target。
MIRFlickr-25K
bit I->T T->I
16 0.915 0.897
32 0.937 0.906
64 0.950 0.915
128 0.961 0.926
NUS-WIDE
bit I->T T->I
16 0.828 0.804
32 0.852 0.823
64 0.867 0.832
128 0.876 0.837
MSCOCO
bit I->T T->I
16 0.905 0.913
32 0.939 0.951
64 0.952 0.964
128 0.958 0.969

### 11.6 正式性能门槛
一级：生存门槛（128-bit 主看）。若低于以下门槛，不允许直接做结构创新结论，必须先回查 Stage 2
/ 3。
- MIR 128-bit：I->T < 0.90 或 T->I < 0.88。
- NUS 128-bit：I->T < 0.79 或 T->I < 0.75。
- COCO 128-bit：I->T < 0.86 或 T->I < 0.88。

二级：达到 RA 对应 bit 长度结果，记为达到目标。三级：同 bit 长度下两个方向平均 mAP 超过 RA，或
一个方向超过 RA 且另一个方向不低于 RA 超过 0.005 以内的容忍带，记为 stretch goal。

### 11.7 一致性门槛
- 128-bit 比 64-bit 低超过 0.02 时必须触发诊断。
- I->T 与 T->I 差值异常放大时必须触发诊断。
- NUS 单方向明显塌缩时必须触发诊断。
- Stage 2 raw CLIP baseline 不差，但 hash mAP 大幅掉队时必须触发诊断。

## 12. 三数据集正式 profile（第一版默认配置）
共享主链不变，profile 可不同。以下 profile 是当前第一版正式默认配置，不是无限开放搜索空间。

### 12.1 MIRFlickr-25K profile
mirflickr25k_profile_v1:
semantic:
lambda_ar_fusion: 0.65
tau_confidence: 0.07
model:
d_z: 256
cheby_order: 4
tree_levels: 2
tree_prototypes: [256, 64]
beta_tree_injection: 1.0
graph_k_train: 15
graph_k_eval: 20
loss:
beta_relation_weight: 1.0
alpha_intra_topology: 0.50
lambda_sem_total: 1.00
lambda_pair_total: 0.60
lambda_q_total: 0.05
lambda_bal_total: 0.01
optim:
lr: 1.0e-4
weight_decay: 1.0e-4
epochs: 120
eval_interval: 5
设计理由：MIR 的文本短、tag 稠密度中等，Stage 1 已做强过滤，因此 direct support A 权重略高，树
层不宜过深。

### 12.2 MSCOCO profile
mscoco_profile_v1:
semantic:
lambda_ar_fusion: 0.75
tau_confidence: 0.07
model:
d_z: 384

cheby_order: 4
tree_levels: 2
tree_prototypes: [384, 96]
beta_tree_injection: 1.0
graph_k_train: 20
graph_k_eval: 30
loss:
beta_relation_weight: 1.0
alpha_intra_topology: 0.60
lambda_sem_total: 1.00
lambda_pair_total: 0.80
lambda_q_total: 0.05
lambda_bal_total: 0.01
optim:
lr: 1.0e-4
weight_decay: 1.0e-4
epochs: 110
eval_interval: 5
设计理由：COCO 多 caption 合并后语义最强，实例级对齐更可靠，因此 lambda_pair_total 高于
MIR。

### 12.3 NUS-WIDE profile
nuswide_profile_v1:
semantic:
lambda_ar_fusion: 0.50
tau_confidence: 0.10
model:
d_z: 256
cheby_order: 3
tree_levels: 3
tree_prototypes: [512, 128, 32]
beta_tree_injection: 1.0
graph_k_train: 30
graph_k_eval: 30
loss:
beta_relation_weight: 1.2
alpha_intra_topology: 0.70
lambda_sem_total: 1.20
lambda_pair_total: 0.40
lambda_q_total: 0.08
lambda_bal_total: 0.02
optim:
lr: 8.0e-5
weight_decay: 2.0e-4
epochs: 160
eval_interval: 10
设计理由：NUS 标签与 tag 恢复噪声更高，结构项 R 更重要，因此 lambda_ar_fusion 下调、
tau_confidence 上调、树层更深、正则更强。

### 12.4 profile 调参边界
Codex 允许调参，但只能在以下有界集合中进行，不允许无限搜索。

- MIR：lambda_ar_fusion ∈ {0.60, 0.65, 0.70}；graph_k_train ∈ {10, 15, 20}；
lambda_pair_total ∈ {0.50, 0.60, 0.70}。
- COCO：lambda_pair_total ∈ {0.70, 0.80, 0.90}；graph_k_train ∈ {15, 20, 25}；d_z ∈ {256,
384}。
- NUS：lambda_ar_fusion ∈ {0.45, 0.50, 0.55}；tau_confidence ∈ {0.08, 0.10, 0.12}；
graph_k_train ∈ {20, 30, 40}；alpha_intra_topology ∈ {0.60, 0.70, 0.80}。
正式要求：超出此有界集合的调参，必须先经你明确批准。

## 13. 更新后的阶段化诊断协议
旧诊断附录的核心秩序是对的；本节把它改成与 Se/C/S 主线一致后的正式版本。

### 13.1 固定逆流排查顺序
1. 先查 Stage 2 baseline：paired vs random cosine gap、raw CLIP I->T/T->I mAP@50。如果这里就
差，先别怪 hash 阶段。
2. 再查 Stage 3 S 的健康度：diag_mean_s、paired_diag_quantile、row/col coverage、NUS
text_source 恢复质量。
3. 再查 Stage 4 到 Stage 6 是否正确消费冻结对象：是否仍然只消费 X_I/X_T/S、是否偷偷回读 raw、
graph scope 是否写错、retrieval inference 是否误走 dense O(N^2)。
4. 最后才调 profile 与损失权重。

### 13.2 NUS-WIDE 专项诊断
- Final_Tag_List.txt 与 All_Tags.txt 是否成功对齐。
- text_source 空行数是否为 0。
- concept_subset 是否冻结。
- image_index.tsv 是否与 raw order 对齐。
- Stage 2 raw CLIP baseline 是否先天偏低。
- Stage 3 row/col coverage 是否单向塌缩。

### 13.3 图构建专项诊断
- train split 是否使用 full-batch formal mode。
- retrieval graph 是否错误走了 dense 全量相似。
- query 与 retrieval 是否误共图。
- FAISS 近邻数是否为空、重复或 coverage 差。

## 14. Codex 执行协议（强约束版）

### 14.1 每次任务的正式任务头
Codex 开始任何实现前，必须先写清：当前 stage、当前子目标、允许修改的文件、禁止修改的文件、
允许新增的文件、预期正式产物、必须通过的校验命令、本轮清理项。

### 14.2 单次任务体量上限
- 新增源文件 <= 8 个。
- 修改源文件 <= 12 个。
- 新增逻辑行数 <= 600 行（不含数据文件、自动生成日志、空行）。
超过上限，必须拆任务。

### 14.3 Git 行为硬约束
- 除非你明确发出“现在提交”的指令，否则禁止 git commit。
- 除非你明确发出“现在提交”的指令，否则禁止 git push。
- 禁止把 data/processed 或 outputs 的产物加入 stage。
- 禁止把 checkpoint、cache、报告、日志当源码提交。

### 14.4 任务结束前的强制清理
1. 删除本轮不再使用的临时脚本。
2. 删除未被正式入口引用的 debug 文件。
3. 清理重复 config。
4. 清理旧 alias。
5. 确保 git status --short 中没有未批准的产物文件。
6. 输出“本轮垃圾清理清单”。

### 14.5 正式 runner 与 validator 优先
任何正式功能必须优先通过 scripts/run_*.py 与 scripts/validate_*.py。禁止只写 notebook、只写交
互 demo、只写单文件临时脚本来冒充正式实现。

## 15. 最终总验收清单
阶段 必须通过的要点
Stage 0 raw 下载/放置完成；raw validator 通过；CLIP

权重可用；environment.lock.json 已写出
Stage 1
manifest 完整；split 闭合；
sample_id_order_sha256 固定；validator 通过
Stage 2
X_I/X_T 正常；meta 正常；validator 通过；
baseline_summary 正常；paired-random gap
为正
Stage 3
A/R/Se/C/S 正常；S 值域正确；diagnostics 正
常；validator 通过
Stage 4-6
formal full-batch train mode 跑通；graph
scope 正确；无 raw 回读；无兜底逻辑；best
checkpoint 可复现
Stage 7
16/32/64/128 bit 全部跑出 I->T/T->I；128-bit 达
到生存门槛；至少一个数据集逼近或超过 RA 参
考目标；报告目录齐全
全局
符号完全统一；无历史别名混入；无垃圾脚本/垃
圾产物污染；Git 状态可审计

## 附录 A：数据来源与原始目录摘要
本附录只给正式来源摘要；实际下载时必须以当时官方页面为准，并把实际文件名、大小、sha256 冻
结进 raw_audit.json。
- MIRFlickr-25K：官方 MIRFLICKR 下载页提供 mirflickr25k.zip 与
mirflickr25k_annotations_v080.zip。
- MSCOCO：官方 COCO 下载页对应 2014 train/val 图像与 annotations_trainval2014.zip。
- NUS-WIDE：官方说明当前重点覆盖 NUS-WIDE.zip 与 NUS_WID_Tags.zip；工程上额外要求本地
image mirror 与 image_index.tsv。

## 附录 B：Codex 单轮任务模板
【当前阶段】
Stage X / 子块名：
【本轮唯一目标】
【允许修改的文件】
-
-

【禁止修改的文件】
-
-
【允许新增的文件】
-
-
【预期正式产物】
-
-
【必须通过的校验命令】
1.
2.
【本轮垃圾清理项】
1.
2.
【汇报要求】
- 修改文件清单
- 核心行为
- 校验命令与结果
- 剩余阻塞点

## 附录 C：正式产物总表
正式产物清单如下，任何未被列入其中的运行中间物，都默认不是正式交付物。
阶段 正式产物
Stage 0
environment.lock.json；raw_audit.json；
raw_validator_summary.json
Stage 1
manifest_raw.jsonl；manifest_filtered.jsonl；
manifest_meta.json；query_ids.txt；
retrieval_ids.txt；train_ids.txt；
preprocess_summary.json；
validator_summary.json；order_hashes.json
Stage 2
X_I.npy；X_T.npy；meta.json；
validator_summary.json；
baseline_summary.json
Stage 3
A.npy；R.npy；Se.npy；C.npy；S.npy；
meta.json；validator_summary.json；
semantic_diagnostics.json；

Omega_topk_diag.npz
Stage 4-6
config_snapshot.json；train_log.jsonl；
epoch_metrics.json；best_checkpoint.pt；
last_checkpoint.pt；derived_supervision/*
Stage 7
metrics_summary.json；
metrics_by_bits.json；top_r_curves.json；
retrieval_examples/*
