# Attention-RNN-Multi-touch-Attribution
Attention-RNN来做多触点归因模型

代码引用自：[jeremite/channel-attribution-model](https://github.com/jeremite/channel-attribution-model)
不过之前作者的库没把数据集放上来，我就看着作者的博客文档，自己造了一个可以跑的数据集。


之前几篇多渠道归因分析应该算是比较通用的一些方法论：
- [多渠道归因分析（Attribution）：传统归因（一）](https://mattzheng.blog.csdn.net/article/details/117290387)
- [多渠道归因分析：互联网的归因江湖（二）](https://mattzheng.blog.csdn.net/article/details/117294925)
- [多渠道归因分析：python实现马尔可夫链归因（三）](https://mattzheng.blog.csdn.net/article/details/117296062)
- [多渠道归因分析（Attribution）：python实现Shapley Value（四）](https://blog.csdn.net/sinat_26917383/article/details/117443680)

之前在查阅资料的时候，有看到一篇更进阶的，用深度学习来解决问题，
论文可参考18年的一篇：
[Deep Neural Net with Attention for Multi-channel Multi-touch Attribution](https://arxiv.org/pdf/1809.02230.pdf)

我们来看这篇以及品鉴一下关联代码：
官方：[channel-attribution-model](https://github.com/jeremite/channel-attribution-model)

我把可以跑通demo代码放在自己的github之中：[mattzheng/Attention-RNN-Multi-Touch-Attribution](https://github.com/mattzheng/Attention-RNN-Multi-Touch-Attribution)

---

@[toc]


---

# 1 基于注意力的循环神经网络多点触摸归因模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705222952376.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)

假设有 如下7-13个 触点的路径：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705221812546.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)



## 1.1 与markov、Sharpley 的差异
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705221919353.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
markov、Sharpley 是市面上最、最常见的两种归因分析的方法了，但是两种都缺少考虑：
- 长路径序列下路径间的影响（markov考虑了顺序）
- 未融入用户属性信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705222009562.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)


## 1.2 注意力的循环神经网络多点触摸归因模型框架
一种基于注意力的循环神经网络多点触摸归因模型，以监督学习的方式预测一系列事件是否导致转换(购买)。
模型可以输出不同节点的重要性（LSTM的），同时还结合了非常关键的信息，将用户背景信息(如用户人口统计和行为)作为控制变量，以减少媒体效应的估计偏差。
来说明几个特色：
- LSTM 来捕捉长路径模式
- RNN with Attention 将时间衰减作为attention加入
- customer profile — embedding layer + ANN：额外融入用户属性信息
- 还可以输出每个触点的重要性（即LSTM的节点）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705222203652.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
用LSTM来解读路径周期，将路径作为input输入LSTM之中
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705222241193.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
将时间衰减作为attention加入

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705222336875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
整个架构图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705222437531.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
左边是路径模块，右边是用户属性模块，
- 路径模块带有attenion（时间衰减）
- 用户模块，属性embedding之后输出
最后两者add()在一起做输出。


## 1.3 LSTM 来捕捉长路径模式
路径变量作为输入数据被发送到LSTM层，并获得每个路径变量的输出。LSTM体系结构能够捕获通道数据的顺序模式。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705224752366.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)


## 1.4 RNN with Attention 将时间衰减作为attention加入
这个环节是节点模块的比较有意思的模块：

> The time-lapse data are scaled and will be used in the revised softmax function in Attention layer.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705224348510.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
在机器翻译上下文中，重复这一步，得到长度等于翻译单词数的输出向量上下文，然后将这些输出再次发送到另一个LSTM中，得到最终的翻译结果。但在本例中，我们只需要从注意力输出一个结果。
值得注意的是，由于时间衰减元素在客户路径中起着作用，我们将修改softmax函数来考虑这个因素。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705225419401.png)

## 1.5 customer profile — embedding layer + ANN：额外融入用户属性信息
一个简单的全连接神经网络来处理客户数据。这部分非常简单，只有几个密集的层。
之前用户编码会用one-hot encoding，这里使用的是embedding layer自训练。

嵌入层 Embedding:将正整数（索引值）转换为固定尺寸的稠密向量。
 例如： `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`
来看一个keras官方的例子[\[Embedding\]](https://keras.io/zh/layers/embeddings/)：
```
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# 模型将输入一个大小为 (batch, input_length) 的整数矩阵。
# 输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
# 现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```


## 1.6 融合层
路径模块和客户属性模块，输出到另一个dense层，然后由sigmoid激活函数到最终，0/1分类

```
 out_att = Dense(32, activation = "sigmoid", name='single_output')(c)

 # Step 3: import embedding data for customer-ralated variables
 input_con,out_control = self.build_embedding_network()
 added = Add()([out_att, out_control])
```
github中的代码两个输出直接相加`add()`，不是`conatenate()`

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705225735742.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
作者自己的测试结果：
- RNN-attenion模型，96% accuracy 和0.98 AUC
- markov模型，0.86 AUC




# 2 下游应用

## 2.1 下游应用一：分配路径权重

我将使用性能最好的模型来计算分配给每个通道的权值。基本上，对于每一个送入模型的观测数据，如果输出概率大于0.5(这意味着该观测将被归类为转换)，我会从Attention层提取权值，并将其累加到相应的通道。
在得到每个渠道的权重后，我们将使用下面的公式来分配营销预算。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021070523004718.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)

## 2.2 下游应用二：预算评估
哪些营销渠道在推动转化率和销售额，意味着你可以更好地将营销资金分配到最有效的渠道上，并更好地跟踪潜在客户的互动
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705224025396.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
如果你有一定预算，你会如何分配；当你通过模型得出不同路径的权重，就可以根据权重来分配。
当然这种方式比较简单，详细可见我之前贴的俩论文：
[预算分配Budget Allocation：两篇论文（二）](https://blog.csdn.net/sinat_26917383/article/details/117713999)


## 2.3 确定购买潜力以及其他更多的变形
如果有顾客点击了很多路径内容还没转化，可以通过模型得到他购买的可能性。

例如，如果你的公司也关心每个渠道其他转化情况(如电子邮件活动中的广告的点击率)，你将在LSTM之上添加更多的层来实现这一点，如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705230344960.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)

此外，您还可以预测一次购买的平均支出或金额，这可能会使分配权重更准确，也可以提供您关于调整供应链的信息。可以参考上文的输出接入：average spending


## 2.4 确定最有影响力的路径
根据每个客户路径的转换概率排名，我列出了最具影响力的N条路径。
代码如下:
```
def critical_paths(self):
    prob = self.model.predict([self.X_tr,self.s0,self.time_decay_tr,self.X_tr_lr.iloc[:,0],self.X_tr_lr.iloc[:,1],
       self.X_tr_lr.iloc[:,2],self.X_tr_lr.iloc[:,3]])
    cp_idx = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)
    #print([prob[p] for p in cp_idx[0:100]])
    cp_p = [self.paths[p] for p in cp_idx[0:100]]
    
    cp_p_2 = set(map(tuple, cp_p))
    print(list(map(list,cp_p_2)))
```


---
# 2  案例代码demo解读
## 2.1 数据样式
在文章[How to implement an Attention-RNN model into solving a marketing problem: Multi-Channel Attribution](https://medium.com/machine-learning-for-business-problem/how-to-implement-an-attention-rnn-into-solving-the-multi-channel-attribution-problem-6fa90d935859)中没有放出数据，所以笔者自己造了按博文自己造了几条，代码可见：[mattzheng/Attention-RNN-Multi-Touch-Attribution](https://github.com/mattzheng/Attention-RNN-Multi-Touch-Attribution)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210705223535431.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)
- total_conv就是最后是否转化
- last_time_lapse，节点访问时间经过，放入的是路径模块，作为attention模块，不过这个样式我自己造的时候也有点不确定，是否每个节点对应，可以自由灵活调配
- marketing_area 、 tier 、 customer_type都是用户属性类型，这里自由发挥，可以很多





---

# 参考文献：
[How to implement an Attention-RNN model into solving a marketing problem: Multi-Channel Attribution](https://medium.com/machine-learning-for-business-problem/how-to-implement-an-attention-rnn-into-solving-the-multi-channel-attribution-problem-6fa90d935859)

[Attention-RNN Channel-Attribution Model](https://jeremite.github.io/RNN-Channel-Attribution-Model/)


