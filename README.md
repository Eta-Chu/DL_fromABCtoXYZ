#DL_fromABCtoXYZ

## 从零搭建神经网络完成简单的图像识别

### 1. 项目目标

+ 初步： 搭建全联接神经网络（无自动微分），完成MNIST数据集（0-9 28*28 黑白（No 3d））的图像识别
+ 中阶：构建张量算子包含自动微分，包括手写优化算法。重写全联接神经网络，完成Fashion MNIST数据集图像识别
+ 高阶：完成卷积神经网络（3D）GPU

### 2. 项目技能

+ C++上外部数据（图像数据集）的I/O
+ *C++线性代数矩阵运算：（初级：Eigen，Xtensor第三方库，高阶：OpenBlas，MKL使用，或者自己实现Matrix，Array类)
+ CMAKE
+ *可微式编程的思想（优化->梯度->可微）
+ *深度学习的基本思路，计算机视觉算法
+ 优化算法（主要是梯度算法优化器的实现）
+ GPU的调用CUDA（如果有可能的话）
+ Git的协同

### 3. 主要参考书目

+ 深度学习入门：基于python的理论与实现
+ python神经网络编程

### 4. 需要的链接
+ MNIST数据集[🔗](http://yann.lecun.com/exdb/mnist/)
+ FashionMNIST数据集[🔗](https://github.com/zalandoresearch/fashion-mnist)
+ Eigen库[🔗](https://eigen.tuxfamily.org/index.php?title=Main_Page)
+ Xtensor库[🔗](https://xtensor.readthedocs.io/en/latest/)
+ OpenBlas[🔗](https://github.com/OpenMathLib/OpenBLAS?tab=readme-ov-file)
+ MKL库[🔗](https://www.intel.cn/content/www/cn/zh/developer/articles/guide/intel-math-kernel-library-intel-mkl-2019-getting-started.html)
