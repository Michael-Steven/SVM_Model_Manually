# SVM支持向量机（使用SMO算法优化）

#### 17373452 单彦博

## 使用说明

将训练集（默认为`svm_training_set.csv`）放在当前目录下，命令行输入：

```shell
$ python svm.py
```

默认将训练数据的$80\%$用来做训练集，$20\%$用来做测试集。可调整训练参数、迭代次数、$C$值、高斯核函数的参数、容错率。训练后的模型保存在`train_result.mat`中。

## 代码功能说明

#### def split_data(input_set):

划分训练集和测试集，默认将训练数据的$80\%$用来做训练集，$20\%$用来做测试集。

#### class SvmModel:

$svm$模型类，用来存储模型的各项数据，如：训练集，标签，$alpha$值，核函数等。

#### def kernel_trans(x, a, sigma):

使用高斯核，根据 $x:$全部数据，和 $a:$某一组的数据，来计算一组数据对应的核函数值。

#### def calculate_p(model, i):

计算第$i$组数据的预测结果。

#### def check_kkt(model, i):

按照规则检查第$i$组数据的$KKT$条件。

#### def update_alpha(model, i):

按照规则更新第$i$组数据的$alpha$值。

#### def smo(data_mat, label, c, eps, iter_num, sigma):

根据输入参数创建$SvmModel$模型，并进行迭代训练。

#### def main():

进行训练数据的划分，归一化，模型初始化，训练，模型保存和加载，测试等操作。