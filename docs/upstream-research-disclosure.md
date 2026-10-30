# AUREL-Mem：研究来源、实现范围与公开披露

本文对应当前发布实现。参考源码审查日期为 2026-10-28；论文标题与作者信息于 2026-10-29 从各论文的 arXiv 页面核对。代码提交标识用于说明研究对象，不表示这些仓库被集成到本项目。

## 实现归属与方法范围

AUREL-Mem 是面向 Add/Search 接口的独立工程实现。开发过程中阅读了下列论文和官方实现，借鉴其设计思路，重新编写了本项目的存储、抽取、检索、排序和证据交付逻辑。发布代码不包含下列参考仓库的源码，也不将它们作为运行依赖。由于研究时阅读过参考源码，本项目不宣称是未经源码接触的 clean-room 实现，也不宣称完整复现任一论文。

本项目的具体组合与工程改动包括：

- 保存有序原始消息与原始图片，结构化记录保留来源关联。图片内容可按哈希复用存储，但不同出现位置保留独立记录，不能仅因文字相同而合并证据。
- 通过 OpenAI 官方 Python SDK 的兼容接口调用 `gpt-4o-mini`，抽取事实、实体、关系、时间、状态、图像描述和 OCR 等字段。查询含图片时先生成检索描述；Search 返回证据，不生成最终问题答案。
- 对原文、结构化语义和视觉/OCR 文字分别进行词法检索，再与稠密向量结果进行 RRF 排名融合。`text-embedding-v4` 请求使用 `dimensions=1024`，返回维度不符会报错。视觉向量来自图像的文字化表示，不是对原始像素进行 CLIP/SigLIP 式联合嵌入。
- 在种子候选上进行有界的会话邻近与实体关系扩展，用互补覆盖奖励和重复惩罚选择证据。实现的是轻量启发式选择器，没有证据森林优化器或独立的模型验证器。
- 保留历史状态证据，单独计算当前状态；区分消息时间、事件时间和写入时间，避免用覆盖旧证据的方式更新记忆。
- 最终按条数、文字和图片预算返回原始图文、来源标识及时间。用户隔离、请求幂等和事务提交用于保证接口一致性。

这些是本项目的实现选择和组合改动，不是对上述通用研究思想的原创权主张。尚未通过消融实验证明各组件带来的效果增益。

## 逐项研究借鉴与未采用范围

| 研究来源 | 采用的设计思路及本项目实现 | 未采用或不能等同的部分 |
|---|---|---|
| [M2A][m2a-paper] | 原始消息层与语义层分离；保留来源关联；融合多个检索排名。 | 未采用 ChatAgent/MemoryManager 双智能体调度、Milvus 存储和论文的数据合成流程；不是 M2A 的服务化封装。 |
| [V-Mem][vmem-paper] | 对文字和图像查询作相应处理；将命中的文字化单元关联回原始消息，并有限扩展同会话上下文。 | 未采用其 MiniLM/SigLIP 双模态索引、完整模态路由策略或全部假想 caption/anchor 流程；图片文字化后检索不等价于原方法。 |
| [ATM-Bench / Schema-Guided Memory（SGM）][atm-paper] | 将 OCR、视觉描述、实体、关系和时间等拆成可检索字段；用 GPT-4o-mini 的结构化抽取整合处理。 | 未采用原实现的独立 EXIF/GPS、OCR 与多次 caption 工具链；不推测未提供的时间、地点；未运行 ATM-Bench 评测。 |
| [GraphMemix][graph-paper] | 种子召回后有界扩展关系邻居；以覆盖和冗余因素选择互补证据。 | 未实现 node verifier、anchor-conditioned ECV、PCSF/MILP 或完整 evidence-forest 优化；不能称为 GraphMemix 复现。 |
| [TAME][tame-paper] | 借鉴稳定事实与短期状态分开管理的思想；本项目保留历史版本并独立计算当前视图。 | 未复制或调用 GPL 仓库代码；未采用其动态事实提升、容量 10 的 FIFO、原地 modify/remove、Grounding DINO、Jina 或完整 RA2G 流程。 |
| [Omni-SimpleMem][omni-paper] | 检索表示与原始证据分离；在输出预算内展开、交付原始材料。 | 未实现其完整金字塔检索、参数记忆、自演化、自动研究或音视频流程；不继承其 LoCoMo / Mem-Gallery 成绩。 |
| [DeliverMem][deliver-paper] | 保留原始模态、可读来源标识与出现时间，使同一来源的文字和图片一起交付。 | 未复现区域多向量召回与像素标识控制；原图交付不能恢复未被召回的线索。研究记录中未获得可下载实现，只参考论文与项目页。 |

## 参考文献与官方代码

1. **M2A: Multimodal Memory Agent with Dual-Layer Hybrid Memory for Long-Term Personalized Interactions**（2026）。Junyu Feng, Binxiao Xu, Jiayi Chen, Mengyu Dai, Cenyang Wu, Haodong Li, Bohan Zeng, Yunliu Xie, Hao Liang, Ming Lu, Wentao Zhang。[论文][m2a-paper]；[官方仓库](https://github.com/Little-Fridge/M2A)。研究提交：`edd8c3b75bae8b2c9c1a0ac8ed67e38c2c2723f8`。
2. **V-Mem: Modality-Routed Retrieval for Long-Term Multimodal Agentic Memory**（2026）。Dingyi Kang, Dongming Jiang, Yi Li, Guanpeng Li, Bingzhe Li。[论文][vmem-paper]；[官方仓库](https://github.com/Dingyi-Kang/V-Mem)。研究提交：`36916b14dc5241e04acbf5cd0f3c40799bc09550`。
3. **According to Me: Long-Term Personalized Referential Memory QA**（2026；ATM-Bench / SGM）。Jingbiao Mei, Jinghong Chen, Guangyu Yang, Xinyu Hou, Margaret Li, Bill Byrne。[论文][atm-paper]；[官方仓库](https://github.com/JingbiaoMei/ATM-Bench)。研究提交：`ef4e5dff1a47ec71213a06e359f02753defa8fb1`。
4. **GraphMemix: Query-Aware Evidence Forests for Long-Term Multimodal Agent Memory**（2026）。Geng Li, Yuhao Wang, Dong Li, Jianye Hao, Yuxin Peng。[论文][graph-paper]；[官方仓库](https://github.com/memorax-ai/graphmemix)。研究提交：`f938a6cebc634d7d8da8971d25d155903c0a52eb`。
5. **TAMEing Long Contexts in Personalization: Towards Training-Free and State-Aware MLLM Personalized Assistant**（arXiv 2025；KDD 2026）。Rongpei Hong, Jian Lang, Ting Zhong, Yong Wang, Fan Zhou。[论文][tame-paper]；[官方仓库](https://github.com/ronpay/TAME)。研究提交：`21d5a7cfa85106f278868e2f60887e8ab264c1f2`。
6. **Omni-SimpleMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory**（2026）。Jiaqi Liu, Zipeng Ling, Shi Qiu, Yanqing Liu, Siwei Han, Peng Xia, Haoqin Tu, Zeyu Zheng, Cihang Xie, Charles Fleming, Mingyu Ding, Huaxiu Yao。[论文][omni-paper]；[官方仓库](https://github.com/aiming-lab/SimpleMem)，研究对象为其中 `OmniSimpleMem/` 实现。研究提交：`db80b6a7c591e0ea730a058e9f5fc4eb06572299`。
7. **Retrieved but Not Delivered: Multimodal Memory Delivery for Long-Term Agents**（2026；DeliverMem）。Yuhang Jiang, Qingwei Liao, Kaize Yin, Xingling Liu, Luca Cuomo, Silvio Bacci。[论文][deliver-paper]；[官方项目页](https://avalon-s.github.io/DeliverMem/)。2026-10-28 的研究记录显示项目页标注 “Code on publication”，未获得代码仓库；本项目没有集成其实现。

## 许可证与代码来源边界

以下信息对应锁定提交的研究快照，不代替上游仓库自己的许可文本：

| 仓库 | 核查到的许可 | 与本发布代码的关系 |
|---|---|---|
| M2A | [MIT LICENSE](https://github.com/Little-Fridge/M2A/blob/edd8c3b75bae8b2c9c1a0ac8ed67e38c2c2723f8/LICENSE) | 只读研究，未打包、导入或移植仓库代码。 |
| V-Mem | [MIT LICENSE](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/LICENSE) | 同上。 |
| ATM-Bench | [MIT LICENSE](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/LICENSE) | 同上。 |
| GraphMemix | [pyproject.toml](https://github.com/memorax-ai/graphmemix/blob/f938a6cebc634d7d8da8971d25d155903c0a52eb/pyproject.toml) 声明 MIT；研究快照根目录无 LICENSE 文件 | 记录声明与独立许可文件缺失的差异；本项目没有集成该代码。 |
| TAME | [GPL-3.0 LICENSE](https://github.com/ronpay/TAME/blob/21d5a7cfa85106f278868e2f60887e8ab264c1f2/LICENSE) | 只作方法与源码研究参考；未复制、修改移植、链接或分发该仓库代码，也不是运行依赖。 |
| SimpleMem | [MIT LICENSE](https://github.com/aiming-lab/SimpleMem/blob/db80b6a7c591e0ea730a058e9f5fc4eb06572299/LICENSE) | 只读研究，未打包、导入或移植仓库代码。 |
| DeliverMem | 未获得可下载代码，未核定代码许可 | 论文及项目页引用，不是软件组件。 |

上表是参考来源的许可记录，不重新许可以上论文或上游代码，也不把 GPL 参考仓库声明为 MIT 组件。运行依赖（如 OpenAI SDK、FastAPI、HTTPX、Pillow 等）仍适用各自许可证；论文引用不替代依赖清单或第三方许可。

## 模型、容量与验证边界

- 模型调用使用 OpenAI 官方 Python SDK，但服务端可为兼容网关；“官方 SDK”不等于“所有模型均由 OpenAI 官方端点提供”。当前配置的模型标识为 `gpt-4o-mini` 和 `text-embedding-v4`，向量维度为 1024；不进行训练或微调，也不下载模型权重。
- 单次 Add 最多 1024 条消息。HTTP 请求体上限 48 MiB。仅接收内联 JPEG/PNG/WebP Base64 图片；单图最多 10 MiB、4000 万像素，不接受动画。Add 或 Search 查询中的图片合计最多 30 MiB。
- `top_k` 范围为 1–100；默认服务检索上限为 32 条，实际条数还受结果数量和交付预算约束。默认最多返回 12 张图片、图片合计 30 MiB 和 100000 个文本字符；达到预算可能省略部分内容并标明，原始存储保留完整证据。部分预算可通过配置调整。
- SQLite 持久化记录；当前查询会读取该用户的索引记录并在进程内计算排名。尚未完成大规模容量、并发或长期稳定性基准，不能据此承诺无限记忆或特定吞吐量。
- 已有接口、原图回传、用户隔离、幂等及视觉/向量链路验证。两个合成图像案例用于检查图片独有 OCR、纯图片查询和仅向量检索，不是正式准确率基准。未运行完整论文基准、未获得赛事官方成绩，也不借用参考论文分数。
- 如演示入口采用 Cloudflare Quick Tunnel，该入口只用于临时联调：隧道重建会更换地址，进程停止后入口失效；本机需要保持运行和联网。官方不保证其可用性，限制为最多 200 个在途请求且不支持 SSE。它不是固定赛事托管地址或长期服务承诺。[官方限制](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)

[m2a-paper]: https://arxiv.org/abs/2602.07624
[vmem-paper]: https://arxiv.org/abs/2608.01543
[atm-paper]: https://arxiv.org/abs/2603.01990
[graph-paper]: https://arxiv.org/abs/2608.26983
[tame-paper]: https://arxiv.org/abs/2512.21616
[omni-paper]: https://arxiv.org/abs/2604.01007
[deliver-paper]: https://arxiv.org/abs/2609.32590
