# VisionTransformer学习笔记

### VisionTransformer原理学习

![img](https://img-blog.csdnimg.cn/20210626105321101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)

模型一共分为3个模块，分别为：

1.Embedding层（Linear Projection of Flattened Patches）

2.Encoder层（Transformer Encoder）

3.分类层（MLP Head）



**Embedding层**

​		为了能将图片像序列一样输入进Transformer，我们首先要把图片分成多个token，这一步我们用一个大卷积核大小并且大步长的卷积核来实现。比如VIT-B/16就是把输入图片输入进一个卷积核大小16x16，步距为16，卷积核个数为768的卷积层来实现。这样会得到一个[14,14,768]维的向量，我们再把前面两个维度展平，得到一个[196,768]维的向量。这样的向量就可以作为一组序列进行输入了，其中196为num_token,768为token_dim。

​		为了进行分类任务，我们要在生成的tokens当中加入一个class token，其token_dim与我们生成的token_dim相同，在我举的例子中为768，所以也就是[1,768]+[196,768]=[197,768]。

​		之后为了让这些序列有位置信息（在图片里也是有位置的），所以我们加入一个Position Embedding，这里的Position Embedding是直接加在tokens上的，并且是一个可训练参数，所以它的维度和之前的要一样，即[197,768]。



**Encoder层**

​		Transformer Encoder的结构就是如下图所示。

<img src="https://img-blog.csdnimg.cn/20210704114505695.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" alt="img" style="zoom:67%;" />

​		其中需要注意的有两点，第一点是Norm层使用的Layer Norm层，这种方法针对每个token进行归一化处理，跟BN层不一样。

​		第二点是MLPblock的第一个全连接层会将节点个数变为输入节点个数的4倍，第二个全连接层再变回去。



**分类层**

​		上面通过Transformer Encoder后输出的shape和输入的shape是保持不变的。

​		这里我们只是需要分类的信息，所以我们只需要提取出[class]token生成的对应结果就行，比如输出结果是[197,768]，我们只需要取出[1,768]放入MLP Head即可。MLP Head原论文中说在训练ImageNet21K时MLP Head是由`Linear`+`tanh激活函数`+`Linear`组成。