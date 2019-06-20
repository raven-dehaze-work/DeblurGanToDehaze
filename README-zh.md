**迁移DeblurGan网络做去雾工作**

- [English Version](<https://github.com/raven-dehaze-work/DeblurGanToDehaze/blob/master/README.md>)
- [中文文档](<https://github.com/raven-dehaze-work/DeblurGanToDehaze/blob/master/README-zh.md>)

DeblurGan论文：https://arxiv.org/pdf/1711.07064.pdf

## 1. 网络结构

Generator

![](https://ae01.alicdn.com/kf/HTB1tOehbAxz61VjSZFrq6xeLFXad.jpg)

Discriminator

![](https://ae01.alicdn.com/kf/HTB1E7evcL1G3KVjSZFkq6yK4XXaa.jpg)

## 2. 训练数据集

训练所用数据集来自 [RESIDE数据集](<https://sites.google.com/view/reside-dehaze-datasets/reside-v0>)

## 3. 去雾效果



![](https://ae01.alicdn.com/kf/HTB1czyDcGWs3KVjSZFxq6yWUXXaB.jpg)

![](https://ae01.alicdn.com/kf/HTB1gSKDcG5s3KVjSZFNq6AD3FXaD.jpg)

![](https://ae01.alicdn.com/kf/HTB1FUWibAxz61VjSZFtq6yDSVXa8.jpg)



### 4. 其它

参考了DeblurGan的实现：<https://github.com/KupynOrest/DeblurGAN>

其余去雾算法：

1. [DCP去雾Matlab实现](<https://github.com/raven-dehaze-work/DCP-Dehaze>)
2. [MSCNN去雾Matlab实现](https://github.com/raven-dehaze-work/MSCNN_MATLAB)
3. [MSCNN去雾Keras实现](https://github.com/raven-dehaze-work/MSCNN_Keras)
4. [MSCNN去雾TensorFlow实现](https://github.com/dishank-b/MSCNN-Dehazing-Tensorflow)
5. [Dehaze-GAN TensorFlow实现](https://github.com/raven-dehaze-work/Dehaze-GAN)



对应博文：https://www.ravenxrz.ink/archives/summary-iv-definition-and-implementation-of-the-final-defogging-scheme-deblur-ganto-dehaze.html