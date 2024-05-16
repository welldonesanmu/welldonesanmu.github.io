---
title: "推荐系统的中 EMBEDDING 的应用实践"
date: 2024-05-13T21:35:16+08:00
lastmod: 2024-5-16T10:35:55+08:00 
draft: false
author: ["Sanmu"] 
comments: true 
tags:
  - Graph embedding            
---

# Word2Vec -- Item2Vec -- Graph Embedding

## Word2Vec

![扫描文稿_page-0001](https://welldonesanmu2.github.io/picx-images-hosting/20240513/扫描文稿_page-0001.32hs5yxpgi.webp)

# Item2Vec

# Graph Embedding

## 1. DeepWalk

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240514/image.4g4bb4dd1g.webp)

2014年的模型，主要思想是在物品组成的图结构上进行随机游走（Random Walk），产生大量物品序列，然后将这些物品序列作为训练样本输入Word2Vec进行训练，得到物品的Embedding。

DeepWalk算法流程：

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240514/image.8s34j09xo3.webp)

![扫描文稿(1)_page-0001](https://welldonesanmu2.github.io/picx-images-hosting/20240514/扫描文稿(1)_page-0001.8ojil4vyoo.webp)

## 2. LINE

Large-scale Information Network Embedding

### 动机

这篇论文的初衷是解决两个问题：

1. **大规模**图节点的表示学习
2. **有向、有权图**的节点表示学习

无监督的，不使用神经网络架构的Graph Embedding技术。LINE 的方法是基于直接优化节点嵌入向量，通过显式定义的一阶和二阶相似性损失函数进行优化，而不涉及神经网络的多层结构或非线性变换。

![扫描文稿(2)_page-0001](https://welldonesanmu2.github.io/picx-images-hosting/20240514/扫描文稿(2)_page-0001.51dyxreuk6.webp)

## 3. Node2Vec

![电话：65642222_page-0001](https://welldonesanmu2.github.io/picx-images-hosting/20240514/电话：65642222_page-0001.3rb1row4ab.webp)

## 4.  EGES

![扫描文稿(3)_page-0001](https://welldonesanmu2.github.io/picx-images-hosting/20240516/扫描文稿(3)_page-0001.7awzkbfdhv.jpg)

[推荐系统中EMBEDDING的应用实践 - 卢明冬的博客 (lumingdong.cn)](https://lumingdong.cn/application-practice-of-embedding-in-recommendation-system.html)

