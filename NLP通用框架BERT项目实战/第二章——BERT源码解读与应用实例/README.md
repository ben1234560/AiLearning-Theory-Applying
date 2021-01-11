# README

[BERT开源框架地址](<https://github.com/google-research/bert>)，最好读下README，以下是预训练好的BERT模型，这里用到两个

![1609825905750](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1609825905750.png)

> 点击即可下载，里面内容如下

![1609826152438](assets/1609826152438.png)

![1609826160982](assets/1609826160982.png)

> json：相关的参数
>
> vocab：语料库
>
> 其它：使用时的是加载文件，如训练好的权重等

![1609826290795](assets/1609826290795.png)

> 下载一个数据集，使用脚本命令下载，可能需要翻墙，可以访问我的云盘进行下载。链接：https://pan.baidu.com/s/18vPGelYCXGqp5OCWZWz36A 
> 提取码：de0f。这里只用到MRPC



#### MRPC

内容如下：

![1609827571649](assets/1609827571649.png)

train.csv：

![1609827493014](assets/1609827493014.png)

> 二分类任务：判断两句话是否说的是同一意思
>
> Quality：是否相同，相同为1

test.csv

![1609827666717](assets/1609827666717.png)

> 没有了Quality，需要进行预测



#### download BERT

把bert的code全部下载下来，并解压到指定目录

![1609827887131](assets/1609827887131.png)



#### 创建环境

~~~
# python3.7，我的是window
pip install tensorflow==1.13.2 -i https://pypi.douban.com/simple

pip install numpy==1.16 -i https://pypi.douban.com/simple
~~~



#### 参数

![1609999902323](assets/1609999902323.png)

注意：是run_classifier.py文件

![1609999954700](assets/1609999954700.png)

~~~
-task_name=MRPC
-do_train=true
-do_eval=true
-data_dir=../GLUE/glue_data/MRPC
-vocab_file=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt
-bert_config_file=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_config.json
-init_checkpoint=../GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_model.ckpt
-max_seq_length=128
-train_batch_size=8
-learning_rate=2e-5
--num_train_epochs=3.0
-output_dir=../GLUE/output/
~~~

> task_name：运行的模块，在main里指定了名字对应的类
>
> do_train：是否训练
>
> do_eval：是否验证
>
> data_dir：数据地址
>
> vocab_file：词库表
>
> bert_config_file：bert参数
>
> init_checkpoint：初始化参数
>
> max_seq_length：最长字符限制
>
> train_batch_size：训练次数
>
> learning_rate：学习率
>
> num_train_epochs：循环训练次数
>
> output_dir：输出路径

配置完成后，run该文件即可

![1610000183063](assets/1610000183063.png)



Google原版的

![1610000131364](assets/1610000131364.png)



#### 报错及解决办法

class AdamWeightDecayOptimizer(tf.optimizers.Optimizer): AttributeError: module 'tensorflow' has no attribute 'optimizers'

> 如下内容

~~~
tf.optimizers.Optimizer改为tf.keras.optimizers.Optimizer
~~~



super(AdamWeightDecayOptimizer, self).__init__(False, name) TypeError: __ini

~~~
super(AdamWeightDecayOptimizer, self).__init__(False, name)
    改成
super(AdamWeightDecayOptimizer, self).__init__()
~~~



tensorflow/core/framework/op_kernel.cc:1401] OP_REQUIRES failed at save_restore_v2_ops.cc:109 : Not found: Failed to create a NewWriteableFile:

> 路径过长，需要将整个项目移动到某盘下。要求满足：1.段路径，2.全英文



#### 备选方案

一直起不来的，可以直接使用我改好的代码文件，链接：https://pan.baidu.com/s/18vPGelYCXGqp5OCWZWz36A 
提取码：de0f

我的路径如下：

![1610161459549](assets/1610161459549.png)

![1610161435131](assets/1610161435131.png)

![1610161444607](assets/1610161444607.png)

![1610161474977](assets/1610161474977.png)

![1610161499158](assets/1610161499158.png)

output是自动生成的不需要管