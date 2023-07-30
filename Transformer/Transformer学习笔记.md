# Transformer学习笔记

### Self Attention原理学习（学习于李宏毅老师的讲解视频）

#### Self Attention是解决什么问题的	

​	Self Attention是解决输入是向量序列问题的，例如，如果输入的是一个句子，那么这个句子中的每个词语都是一个词向量，如果输入的是一段音频，这里边每一个小的输入就是一小段音频，这一小段音频也就是一个向量。

#### Self Attention的输出

​	Self Attention的输出有三种形式：

​			1.输出与输入向量数目相同，即每一个向量在输出时都会对应一个标签，像词性分类，我们会给每个词语输出一个词性标签。	

​			2.整个输入的序列组只输出一个结果，像句子的情感判断，我们输入一个句子，最后输出的是这个句子是好是坏的标签。

​			3.我们不知道要输出多少个标签，让机器自行决定。例如seq2seq，翻译的时候一种语言的输入数量和另一种语言的输出的数量没关系。

#### Self Attention的计算

​		Self Attention是一个计算某个向量与其他所有向量关联度的过程 

<img src="./img/image-20230728110049977.png" alt="image-20230728110049977" style="zoom:67%;" />

如上图所示，a就是我们要处理的向量组，假如我们要计算a1与其他向量的关联性，我们首先用W<sup>q</sup>

与a1相乘得到q<sup>1</sup>，q称为Query，有关键字的意思 ，之后所有的a（包括自己），分别乘W<sup>k</sup>，得到各自的key，之后q<sup>1</sup>会与所有的key分别进行点乘（Dot-Prouduct），就会得到a1与其他所有a之间的关联度α（称为Attention Score），之后会进行一个softmax（这里也可以用其他函数，例如ReLu），得到α<sup>’</sup> 。



<img src="./img/image-20230728110253039.png" alt="image-20230728110253039" style="zoom:67%;" />

之后，每一个a会计算出一个v，这个v相当于这个向量所包含的信息，之后让我们计算出的α<sup>’</sup>与v相乘，这个就是一个如果我和你关联性大，我就从你那里多抽取一些信息的过程，之后再将从每个a那里抽取出来的信息相加，得到a1对应的b1。



​	事实上，上述所说的过程，在实际计算时，b1，b2，b3，b4是同步计算过来的，也就是说前面的各个a的q，k，v，α也是同步计算来的，这里用到了矩阵乘法，运算如下图所示。

<img src="./img/image-20230728111637061.png" alt="image-20230728111637061" style="zoom:50%;" />



<img src="./img/image-20230728111729977.png" alt="image-20230728111729977" style="zoom:50%;" />

<img src="./img/image-20230728111845553.png" alt="image-20230728111845553" style="zoom:50%;" />



### Multi-head Self-attention原理学习

#### 为什么有Multi-head Self-attention

​		在Self Attention中，其实就是在找q与k的相关，但是相关是可以有很多种不同的定义的，所以我们可以计算出多个q，让这些不同的q负责不同种类的相关。

#### Multi-head Self-attention的计算

Multi-head Self-attention的计算与Self Attention很类似，比如你有2个头，那就让q，k，v分别乘以两个矩阵，得到q1，q2，k1，k2，v1，v2，然后q1，k1，v1为一组计算Self Attention得到b1，另一组计算出b2，之后将b1和b2合起来乘一个矩阵得到最终b。

​	<img src="./img/image-20230728120946549.png" alt="image-20230728120946549" style="zoom:50%;" />



### Transformer原理学习

#### 模型结构

​		首先，要说明的一点是Transformer是一个seq2seq结构，seq2seq结构的最大特点是输入与输出的长度都是不确定的，就像最上边说的Self Attention的输出结果的第三种。

​		接下来解释Transformer的结构，Transformer由Encoder和Decoder组成。

##### Encoder

​		首先来看Encoder部分。

<img src="./img/image-20230728124814171.png" alt="image-20230728124814171" style="zoom: 50%;" />

​		Encoder部分由n个上述两层结构组成，其中在第一层首先经过一个Multi-Head Attention，之后会通过一个残差边与原本输入相加，最后经过一个归一化层。

​		在第二层首先经过一个Feed Forward层，之后再通过一个残差边与原本输入相加，最后经过一个归一化层。

##### Decoder

​		接下来来看Decoder结构。

<img src="./img/image-20230728145843615.png" alt="image-20230728145843615" style="zoom: 67%;" />

​		首先来说明上图结构的输入，这个结构的输入是一个随着你不断输出而不断增加的过程，比如一开始什么都没有，就只有一个BOS的符号，这时候就只输入一个BOS的符号，之后呢，Decoder根据你的输入输出了“机”，那么接下来BOS符号和“机”就会一起再次输入进Decoder用来预测第二个字，依此类推，直到最终Decoder输出了END符号，整个预测才会结束。也就是说，Decoder在预测每一个字的时候，都是结合了之前的所有预测结果来进行预测的。下图是一个形象的解释。

<img src="./img/image-20230728150915143.png" alt="image-20230728150915143" style="zoom:50%;" />

​		之后来说明Decoder的结构构成，它的结构与Encoder相差不多，一共有两点区别，第一点是出现了一个Masked Multi Head Attention，第二点是中间的Multi Head Attention当中有两个信息是从Encoder的输出过来的。

​		首先来说明第一点不同，为什么使用Masked Multi Head Attention，因为前面说到，在预测当前的输出时，只能使用到之前的输出作为输入，所以为了防止模型偷看到后边的输入，我们用一个mask把后面挡住。（这里其实我一开始看到的时候很疑惑，明明上边在讲过程时，是一个个生成的，不应该能看到后边的东西才对，比如说我预测“器”这个字的时候，我所拥有的输入应该只有BOS以及“机”这个字，“学”和“习”明明还没生成，那为啥还需要mask呢。   在后边的学习我才知道，在训练模型时，会使用一个叫做teaching forcing的技术，这个技术就是说，你在训练时，直接把Ground truth当做输入进行输入，这样的话，就会直接把机器学习这四个字输入进来，这时候就有了使用mask的作用，只需要依次减少mask就可以达到上述原理地方所讲的过程）

​		接着说明第二点不同，中间的Multi Head Attention中的两个输入是由Encoder的输出来的，这两个输入会作为k和v，自己这边的这个输入会作为q，之后进行Self Attention的计算，这个过程称为Cross Attention，流程如下图所示。

<img src="./img/image-20230728153142560.png" alt="image-20230728153142560" style="zoom:50%;" />

