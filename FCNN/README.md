# 全连接神经网络（Fully Connected Neural Network FCNN）

## 1.矩阵操作
矩阵操作可以被大致分为以下几个部分：
### (1). 基本矩阵运算：
    基本矩阵运算（包括向量运算）极其范式化，并且细节非常底层，所以在BLAS[🔗](https://www.netlib.org/blas/)完成这一目标之后，几乎所有涉及到矩阵运算的地方（比喻说不同的第三方库Eigen，Xtensor，python的numpy以及matlab等等）都只是做了数据的类型转换然后调用BLAS库而已。所以可以理解如果仅仅是做矩阵运算，通过不同的方式得到的速度都是一样的，区别只在于调用函数和数据转换的速度。
    在BLAS里面，它将矩阵运算分为三个层面
    + vv(vector-vector)：向量的数乘运算，以及向量间的点乘，加法，以及向量的模运算。
    + mv(matrix-vector)：矩阵和向量之间的乘法运算。
    + mm(matrix-matrix)：矩阵和矩阵之间的乘法运算。
涉及到所有的矩阵向量以及标量数乘运算。

### (2). 线性代数运算：
指的是基于矩阵运算的一些操作，比喻说
    + 矩阵分解（奇异值分解，QR分解，LU分解）
    + 特征值问题求解
    + 线性方程组求解
    + ...
这些运算涉及到的线性代数算法也已非常成熟，在Lapack[🔗](https://www.netlib.org/lapack/)库中得到了实现，所以几乎所有的涉及这方面的其余第三方库都是调用的Lapack库。

### (4). 函数运算:
函数运算即对矩阵所有元素的函数运算，这些一般在高级矩阵库中都有对应的函数实现。也可以基于BLAS自己实现。

### (3). 视图操作：
在现代面向对象的高级矩阵库中，一般都会有视图操作。在这些库中，矩阵或者高维数组都是属于一个抽象类，其数据都是连续存储在内存上的数，所以涉及到很多列操作和行操作时都是指针和数在数组中的位置的映射操作，这些操作这里统称为视图操作，大致有以下
+ 指标操作：按照行列指标取具体位置的数例如：M[i,j]
+ 切片操作（slice/block）：取某几行某几列或者某一块的数例如：M[i:i+2,j:j+3]
+ 转置操作：矩阵的转置操作
+ 反转操作等等。。。
需要注意的是这些操作一般都是返回一个新的数组的抽象类，而其数据的地址还是跟原始抽象类共享的，所以在此类操作的时候需要注意深拷贝和浅拷贝的区别。

这里介绍所有需要用到的库：
+ BLAS[🔗](https://www.netlib.org/blas/)：基本矩阵运算标准库
+ Lapack[🔗](https://www.netlib.org/lapack/)：基于BLAS矩阵运算的线性代数库，本项目不涉及
+ OpenBLAS[🔗](https://github.com/OpenMathLib/OpenBLAS?tab=readme-ov-file)和MKL[🔗](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2024-1/overview.html)：前面说过BLAS几乎是计算机矩阵运算的标准库，几乎没有人会再从零写矩阵基本运算。但是基于不同的硬件计算平台会有优化，OpenBLAS和MKL就是做了硬件的优化计算，MKL是专门针对Intel芯片的优化。一般来说人们不会再调用BLAS，而是针对不同的硬件选择MKL或者OpenBLAS。
+ Eigen和Xtensor就是现代化的矩阵运算库，包含以上四种所有运算。其库的实现思路基本上都是实现一个抽象的矩阵类，然后涉及到具体的运算调用BLAS（OpenBLAS，MKL）以及Lapack库。所以在使用这些库的时候一般需要指定它矩阵运算link到的具体的运算库，一般默认的都是OpenBLAS库。


## 2. 涉及到的运算函数汇总
+ 对于Eigen的API建议快速阅读这两个部分[1](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)和[2](https://eigen.tuxfamily.org/dox/AsciiQuickReference.txt)


|具体操作|Eigen中的函数|Xtensor中的函数|
|:--------|:-------------|:--------------|
|生成随机矩阵|[MatrixXf::Random()](https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae814abb451b48ed872819192dc188c19)||
|生成零矩阵|[MatrixXi.Zero()](https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#a422ddeef58bedc7bddb1d4357688d761)||
|找最大值|M.maxCoeff()||
|指数运算|M.array().exp()||
|求所有数的和|M.sum()||


