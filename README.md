# 《基于Deep Learning的Massive MIMO-MMCM解调技术》——一种免信道估计方法
README
===========================
作者 masxo 2018/03/17 
</br>这是MIMO-MMCM的Python库 </br>
</br>MIMO-MMCM的核心思想是利用Chirp信号的抗多普勒抗多径等复杂信道环境等特点，利用深度学习算法构建解调模型，另一个核心思想是在扩频信号的时带宽积足够大的情况下可免于信道估计。2018年以前的DeepLearning-based MIMO检测论文都是基于已知信道状态信息用来训练深度学习模型。</br>
</br>而DeepLearning-based MIMO-Chirp充分利用了Chirp信号特点，离线状态下不需要知道CSI便可训练出理想的DL模型。 </br>
</br>这个库的DL MIMO-MMCM模型将信源归为整数型量，信源取值范围由QAM调制阶数QAM_M、MMCM子带个数M、子带时带宽积P、信源组个数J、发射天线个数TxN、接收天线个数RxN共同决定，本程序中不使用空时编码。</br>
</br>此外，DL模型采用TensorFlow搭建</br>

****

Python库依赖
----------
```Python
import MIMO_MMCM_PyLib as MC
import numpy as np
import tensorflow as tf  
```






