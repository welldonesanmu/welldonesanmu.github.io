---
title: "GraphSAGE"
date: 2024-05-11T15:18:34+08:00
lastmod: 2024-05-11T15:18:34+08:00 
draft: false
author: ["Sanmu"] 
comments: true 
tags:
  - GNNs            
---

# GraphSAGE

Inductive Representation Learning on Large Graphs

两点强调：

- Inductive
- Large Graphs

首先明确一个概念：在推荐系统中Graph embedding采用直推式或称推导式（**transductive**）和归纳式（**Inductive**）两种，区别就是对于一个新加入的节点，模型是否能够在不重新学习之前预训练好的embedding的情况下，对这个新加入节点的特征进行嵌入表征。

这个概念比较老，面试时不要被迷惑。

|            |                                      |
| :--------- | ------------------------------------ |
| **直推式** | **基于矩阵分解、DeepWalk、Node2Vec** |
| **归纳式** | **GraphSAGE、 GCN、 GAT、 GCLs**     |



- 直推式这类方法通常包括直接优化节点嵌入的方法，如DeepWalk、Node2Vec等。这些方法在一个特定的图上学习节点的嵌入向量，并且假设训练时使用的所有节点在预测时都是可见的。这些方法的共同点是它们通常需要对图中所有节点进行处理，并且一次学习所有节点的嵌入。因此，它们通常不具备归纳能力，即难以直接应对图中新增节点的情况，除非重新进行嵌入的学习过程。
- 对于GCN是哪种范式一下就明了了，GCN通过学习一个卷积过程，其中节点的特征信息是通过其邻居的特征进行聚合更新的。这个过程允许GCN处理未见过的节点，所以是归纳式。但GCN的归纳能力取决于它的应用方式和具体的训练设置。例如，如果GCN在一个特定的图结构上训练，并且仅用于相同结构的图，则其行为更倾向于转导式学习。但如果其训练过程涵盖了多种图结构，并且模型设计上允许它适应不同的图，那么它就显示出强大的归纳能力。（**多图案例是蛋白质结构预测**）
- GCLs通常在无监督或自监督的设置下用于学习图或节点的表示。这种方法通过最大化正样本对的相似性和最小化负样本对的相似性来学习有效的嵌入。图对比学习的关键优势在于它不依赖于标签数据，而是通过数据本身的结构和内容来学习表示。这个也是归纳式的。

---

**GraphSAGE**（Graph Sample and Aggregation）是一种归纳式图神经网络模型，它设计用来生成节点的嵌入，**即使是在训练过程中未见过的节点**（这一点很好保证：1训练时隐藏节点2邻居采样）。这种归纳能力源于其独特的聚合机制，该机制能够从节点的局部邻域信息中学习如何有效地生成嵌入。

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.39kzyhot2c.webp)

算法流程：

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.7p3f3rhtsv.webp)

GraphSAGE实现归纳式学习的关键方面：

### 1. **局部邻域聚合**

GraphSAGE的核心思想是使用聚合函数来合成一个节点的邻居信息，形成该节点的嵌入。这意味着生成节点嵌入不依赖于整个图的结构，而是依赖于每个节点的局部邻域。具体聚合函数可以是简单的均值聚合、池化聚合或LSTM聚合等。通过这种方式，模型可以灵活地应对图中的新节点，因为它只需要新节点的局部信息就能计算其嵌入。

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.3rb1n2m3d8.webp)

![image](https://welldonesanmu2.github.io/picx-images-hosting/20240511/image.5j40hzmpae.webp)

### 2. **邻居采样**

由于实际图往往非常大，直接使用所有邻居的信息进行聚合计算是不现实的。GraphSAGE引入了邻居采样策略，即从每个节点的邻居中随机选择一个固定大小的子集，然后只使用这些采样邻居的信息来进行聚合。这不仅减少了计算负担，而且通过随机性增加了模型的泛化能力。

### 3. **多层聚合**

GraphSAGE通常使用多层聚合结构，每层都对应一个聚合步骤。每个节点在每一层聚合其邻居的信息，然后将聚合结果传递到下一层。这样，更高层的聚合可以间接包含更远邻居的信息。这种分层的聚合方式允许模型捕获从近邻到远邻的结构信息。

---

TensorFlow实现：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
参考文献:
    [1] Hamilton W, Ying Z, Leskovec J. Inductive representation learning on large graphs[C]//Advances in Neural Information Processing Systems. 2017: 1024-1034.
    (https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

class MeanAggregator(Layer):
    # 平均聚合层，用于GraphSAGE中的邻居信息聚合
    def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
                 use_bias=False, seed=1024, **kwargs):
        super(MeanAggregator, self).__init__()
        self.units = units  # 输出单元数
        self.neigh_max = neigh_max  # 最大邻居数
        self.concat = concat  # 是否拼接
        self.dropout_rate = dropout_rate  # Dropout比率
        self.l2_reg = l2_reg  # L2正则化系数
        self.use_bias = use_bias  # 是否使用偏置
        self.activation = activation  # 激活函数
        self.seed = seed  # 随机种子
        self.input_dim = input_dim  # 输入特征维度

    def build(self, input_shapes):
        # 构建层的权重和偏置
        self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=glorot_uniform(seed=self.seed),
                                             regularizer=l2(self.l2_reg),
                                             name="neigh_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), initializer=Zeros(), name='bias_weight')
        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        # 层的前向传播
        features, node, neighbours = inputs
        node_feat = tf.nn.embedding_lookup(features, node)  # 获取节点特征
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)  # 获取邻居特征

        # 应用dropout
        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)

        # 拼接并计算平均值
        concat_feat = tf.concat([neigh_feat, node_feat], axis=1)
        try:
            concat_mean = tf.reduce_mean(concat_feat, axis=1, keep_dims=False)
        except TypeError:
            concat_mean = tf.reduce_mean(concat_feat, axis=1, keepdims=False)

        # 使用权重计算输出，并可选加偏置
        output = tf.matmul(concat_mean, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        return output

    def get_config(self):
        # 返回层的配置
        config = {'units': self.units, 'concat': self.concat, 'seed': self.seed}
        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PoolingAggregator(Layer):
    # 池化聚合层，支持平均池化或最大池化
    def __init__(self, units, input_dim, neigh_max, aggregator='meanpooling', concat=True,
                 dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0, use_bias=False, seed=1024):
        super(PoolingAggregator, self).__init__()
        self.output_dim = units
        this.input_dim = input_dim
        self.concat = concat
        self.pooling = aggregator
        this.dropout_rate = dropout_rate
        this.l2_reg = l2_reg
        this.use_bias = use_bias
        this.activation = activation
        this.neigh_max = neigh_max
        this.seed = seed

    def build(self, input_shapes):
        # 定义层的权重和内部的密集层
        self.dense_layers = [Dense(self.input_dim, activation='relu', use_bias=True, kernel_regularizer=l2(self.l2_reg))]
        self.neigh_weights = self.add_weight(shape=(self.input_dim * 2, this.output_dim),
                                             initializer=glorot_uniform(seed=self.seed),
                                             regularizer=l2(this.l2_reg),
                                             name="neigh_weights")
        if self.use_bias:
            this.bias = self.add_weight(shape=(this.output_dim,), initializer=Zeros(), name='bias_weight')
        this.built = True

    def call(self, inputs, mask=None):
        # 层的前向传播
        features, node, neighbours = inputs
        node_feat = tf.nn.embedding_lookup(features, node)  # 获取节点特征
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)  # 获取邻居特征

        # 重塑并应用密集层到邻居特征
        dims = tf.shape(neigh_feat)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(neigh_feat, (batch_size * num_neighbors, this.input_dim))
        for l in this.dense_layers:
            h_reshaped = l(h_reshaped)
        neigh_feat = tf.reshape(h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))

        # 根据指定的聚合类型应用池化
        if this.pooling == "meanpooling":
            neigh_feat = tf.reduce_mean(neigh_feat, axis=1, keep_dims=False)
        else:
            neigh_feat = tf.reduce_max(neigh_feat, axis=1)

        # 拼接节点和邻居特征，计算输出
        output = tf.concat([tf.squeeze(node_feat, axis=1), neigh_feat], axis=-1)
        output = tf.matmul(output, this.neigh_weights)
        if this.use_bias:
            output += this.bias
        if this.activation:
            output = this.activation(output)

        return output

    def get_config(self):
        # 返回层的配置
        config = {'output_dim': this.output_dim, 'concat': this.concat}
        base_config = super(PoolingAggregator, this).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def GraphSAGE(feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu,
              aggregator_type='mean', dropout_rate=0.0, l2_reg=0):
    # 构建GraphSAGE模型
    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int32)
    neighbor_input = [Input(shape=(l,), dtype=tf.int32) for l in neighbor_num]

    # 根据指定的聚合类型选择聚合函数
    if aggregator_type == 'mean':
        aggregator = MeanAggregator
    else:
        aggregator = PoolingAggregator

    h = features
    # 构建聚合层的多层结构
    for i in range(0, len(neighbor_num)):
        if i > 0:
            feature_dim = n_hidden
        if i == len(neighbor_num) - 1:
            activation = tf.nn.softmax
            n_hidden = n_classes
        h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                       dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type)(
            [h, node_input, neighbor_input[i]])

    output = h
    input_list = [features, node_input] + neighbor_input
    model = Model(input_list, outputs=output)
    return model

def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True):  # 为节点抽样邻居
    _sample = np.random.choice
    neighs = [list(G[int(node)]) for node in nodes]  # 获取每个节点的邻居
    if sample_num:
        if self_loop:
            sample_num -= 1  # 如果包含自环，则调整抽样数目

        samp_neighs = [
            list(_sample(neigh, sample_num, replace=False)) if len(neigh) >= sample_num else list(
                _sample(neigh, sample_num, replace=True)) for neigh in neighs]  # 抽样邻居
        if self_loop:
            samp_neighs = [samp_neigh + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # 包含自身

        if shuffle:
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]  # 可选的邻居打乱
    else:
        samp_neighs = neighs  # 如果未指定抽样数目，使用所有邻居
    return np.asarray(samp_neighs, dtype=np.float32), np.asarray(list(map(len, samp_neighs)))  # 返回抽样邻居和它们的数目

```

