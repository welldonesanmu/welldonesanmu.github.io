---
title: "SDGCL"
date: 2024-05-11T17:59:03+08:00
lastmod: 2024-05-11T17:59:03+08:00 
draft: false
author: ["Sanmu"] 
comments: true 
tags:
  - GCLs            
---

# SDGCL

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.4xucvshemo.webp)

核心思想和主要贡献是提出了一个新颖的自监督学习框架——逐步扩散图对比学习（Stepwise Diffusion Graph Contrastive Learning, SDGCL），用于改进图上的节点分类性能。这个框架主要通过一个**逐步扩散 (stepwise diffusion)** 的过程生成多个增强视图，这一过程利用了单个神经网络的输出，从而能更有效地捕捉节点之间的复杂交互关系，同时减少了内存和计算负担 (使用一个参数矩阵这和堆叠多层的GraphSAGE等GNN模型相比，减少了n-1个W)。此外，这个框架还设计了一个逐步对比损失（stepwise contrastive loss），通过整合所有扩散视图中正负样本的比较，进一步提高了节点嵌入的区分度。

---

### 对比视图的Intuition

Graph通常都是基于图同质性假设，但是异构图balabala。所以我们在一次call / forward中加入stepwise diffusion（本质是change A） 之后生成的中间embedding append并将这些中间层作为负样本，这样只需single layer就可以完成K-hop的长距离信息捕获（消融实验）。

---

### Stepwise Contrastive Loss 设计

利用逐步扩散过程，从单一神经网络输出生成多个增强视图。对于每一个节点，其在不同视图中的表示构成正样本对，而与其他节点在相同或不同视图中的表示构成负样本对。对每一对节点嵌入（来自同一节点但位于不同扩散视图中），计算它们之间的相似度，并与其与其他节点（负样本）的相似度进行比较。

在讨论“逐步对比损失（Stepwise Contrastive Loss）”之前，我们先理解它所依据的基本原理：对比学习。对比学习主要是通过最大化相似（正）样本对之间的一致性，并最小化不相似（负）样本对之间的一致性来训练模型，以此学习数据的有效表示。

以下是具体实现：

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.8hgalmmzka.webp)

### 与InfoNCE Loss和NT-Xent的区别

- 各部分Loss加权归一化后求和

- Pivot view选择第一个视图，对于每一个视图来说（diffusion之后的embedding），把S步内的所有node也当做正样本（Future Work）！

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.6pnbqqk8ig.webp)

### 两篇的区别？

第一篇发现了这个特点之后，做了简单的实验发现结果也不错，使用多层堆叠的形式来生成多个对比视图（one layer one view）（自适应dropout，long range聚合进来的特征比例要小）

### Graph Diffusion技术是怎么实现的？

**Graph Diffusion** 模拟信息在图中的传播来捕获节点之间的长距离关系。

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.8vmqcn6foy.webp)

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.6wqjmb1vqc.webp)



