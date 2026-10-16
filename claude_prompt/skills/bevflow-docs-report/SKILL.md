---
name: bevflow-docs-report
description: 写 BEVflow 仓库里的 docs 和实验报告时使用。覆盖文档分类结构，报告模板，简要写作规则。
---

# 目标

文档简短，分类存放，读者一分钟内找到结论。`$CODE` 里只放简要 docs，长日志和大表放 `$RESULTS`，文档里给路径。

# 分类结构

```
docs/
├── README.md        索引，一句话说明每个子目录
├── setup/           recon  upstream  upstream_patches  env  resources  smoke
├── design/          architecture  upstream_map  interfaces  reproduce_<name>
├── experiments/     howto  registry  protocols
├── reports/         R1  reproduce_<name>
└── reference/       param_card  baseline_numbers
```

每篇不超过两页。超过就拆。

# 报告模板

1. 目的：这次回答什么问题，一句话。
2. 设置：数据版本，划分，配置文件，参数卡链接，种子，硬件。
3. 结果：一张表。数字来自 `metrics.json`，脚本生成。表里放参考值，注明来源与"转引，未核对"。
4. 偏差：与参考值的偏差，以及已排查的原因。
5. 结论：测出来的写"我们观察到"，没测的写"我们假设"并标【假设】。
6. 局限与下一步。

# 写作

若装有 `kaiming-style` skill，先加载它，以它为准。没有就按这几条：

- 问题优先，一句话讲清核心，再展开。
- 短句，主动语态，一句不超过 40 字。
- 用具体的词和数字，不写"显著""强大"。
- 事实和假设分开。
- 同一个概念全文只用一个名字。
- 每段第一句是结论。
- 不用破折号和分号。括号，顿号，冒号都克制。
