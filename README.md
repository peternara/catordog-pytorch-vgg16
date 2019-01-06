# catordog
python pytorch vgg16 resnet50
猫狗分类问题是机器学习领域的经典问题，也是进行深层次研究的必由之路。通过解决猫狗分类问题，我们可以深入理解机器学习的分类方法，掌握分析和解决问题的一般思路，巩固课上所学知识，学以致用。 本文对猫狗分类问题进行了深入的研究，首先对数据集进行扩充和整合，搜索互联网上猫狗数据集，并选择性的补充到训练数据集中。鉴于传统分类方法的局限性，我们采用自己搭建神经网络、VGG16网络模型、ResNet50网络模型三个模型，对每种模型的优缺点进行分析，对模型训练结果进行评估，力求扎实掌握神经网络的构造方法，熟悉和掌握模型的迁移使用，了解最先进的神经网络模型和深度学习最前沿知识。 文章结尾，我们对上述三个模型的结果进行了对比分析，同时对猫狗分类问题进行了未来展望，提出了接下来需要解决的问题。

本实验提供的训练数据是8000条，测试数据是2000条。由于训练数据较少，我们实验的第一步就是对训练数据集进行扩充。通过查找互联网和相关资料，我们获得了25000条训练数据，加上原有8000条，我们的训练数据集规模为33000条。train 训练集包含了 33000 张猫狗的图片，猫狗各一半，每张图片包含图片本身和图片名。命名规则根据 “type.num.jpg” 方式命名。

我的程序使用两个大小为8127MB的GPU。epoch数值选取了30，batchsize为150。训练结束后，我们将模型保存在文件dog_cat_model_resnet50.pkl中，方便进行验证和预测。


自己构建网络 73.95%
VGG16 98.35%
ResNet50 96.6%
