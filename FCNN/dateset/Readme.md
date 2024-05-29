> notice:

    `train` 为训练集数据

    `test` 为测试集数据
    
    `labels` 为标签，也就是准确的目标值
    
    `image` 为具体的图片数据

> 分成四个部分load进来为Eigen::MatrixXd的格式

> images应该load为$n*784$的矩阵，n为样本数量，$784=28*28$

> labels原始为n的一维向量，每一个元素为0-9之间的整数，标记为每一个样本对应的数字
> 应将labels转化为n*10的独热编码(one-hot)的矩阵，其中每一行中只有对应数字的位置为1其余为0
