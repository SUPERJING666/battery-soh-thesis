# 论文代码与数据归档说明

本归档对应《融合人工与机器学习特征的锂电池健康状态估计方法研究》评阅修改黄色高亮版，采用 corrected_v3_connected_static_inputs 协议。固定划分为 39/13/14 个电池、13762/4448/4040 条循环；模型与输入方案见 README 和 configs/protocol.json。

- data：已处理的两份数据；不是原始采集数据，151点序列没有原始时间戳。数据哈希另见 reports/data_manifest.csv。
- results 和 literature_results：现有修正实验的检查点、变换器、逐样本预测和结果；具体完成状态以各 result.json 为准。文献对照不是原论文代码的严格复现。
- archive/submission：原送审版代码、笔记本及 H5 权重的部分快照，旧版数据、预测和其他结果未收录；source_manifest.json 是原始完整快照的来源记录，并非本仓库文件清单。
- archive/training_code_v1、training_code_v2：历史训练代码快照；历史实验输出未收录，不能将历史结果和当前版本混用。
- 不包含虚拟环境、依赖安装包、旧版大型输出、日志或论文正文。prepare.py 依赖原工作区，独立检出无需运行。
- 文件 SHA-256 和字节数见 FILE_MANIFEST.csv；该清单不包含自身。仅加载可信来源的 pickle 和模型文件。

运行依赖和实验限制详见 README.md、requirements.lock.txt 与 reports。
