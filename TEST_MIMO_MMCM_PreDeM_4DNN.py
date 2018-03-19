import MIMO_MMCM_PyLib as MC
import numpy as np
import tensorflow as tf
import tensorboard as tb
import matplotlib.pyplot as plt

# 初始化参数
# SysPara [P,M,J,QAM_M,TxN,RxN]
P = 64
M = 1
J = 1
QAM_M = 4
TxN = 8
RxN = 8
N = P*M
# MAXData = M*J*QAM_M*TxN
MAXData = M*J*QAM_M*TxN
MAXTrainSamplePerData = 100
MAXTest_SamplePerData = 10
MAXTrainSample = MAXTrainSamplePerData * MAXData # 训练总样本数是 MAXSample * MAXData
MAXTest_Sample = MAXTest_SamplePerData * MAXData # 测试总样本
print('max input data is ', MAXData)
MC.InitMIMO_MMCM_PyLibSysPara([P, M, J, QAM_M, TxN, RxN])
Cj = MC.Gen_Cj()
pinvHc = MC.Calc_pinvHc(Cj)
# 生成训练样本标签label_train,生成训练样本标签训练数据，images
label_train = np.zeros((MAXTrainSample, MAXData), dtype=float)
image_train = np.zeros((MAXTrainSample, 2*N*RxN), dtype=float)
for datcut in range(0, MAXData):
    aMj = MC.Sour2aMj(datcut)  # 信源映射
    TxSig = MC.MIMO_MMCM(Cj, aMj)  # 调制
    for SampleCut in range(0, MAXTrainSamplePerData):  # 每个数据生成10万个样本
        label_train[SampleCut + datcut * MAXTrainSamplePerData, datcut] = 1
        # MIMO信道这里是平坦衰落信道,后续添加多径多普勒衰落信道
        H = np.random.rand(RxN, TxN) + 1j*np.random.rand(RxN, TxN)
        H = (H + np.eye(TxN))/4
        # H = np.eye(RxN, TxN)
        noise = (np.random.rand(RxN, 1) + 1j*np.random.rand(RxN, 1))/10
        RxSig = np.dot(H, TxSig)# 接收
        # 重组信号
        ReshapeRxSig = np.append(np.real(RxSig.reshape(-1)), np.imag(RxSig.reshape(-1)))
        image_train[SampleCut + datcut * MAXTrainSamplePerData, :] = ReshapeRxSig
plt.plot(ReshapeRxSig)
plt.show()
# 生成测试样本标签label_train,生成测试样本标签训练数据，images
label_Test = np.zeros((MAXTest_Sample, MAXData), dtype=float)
image_Test = np.zeros((MAXTest_Sample, 2*N*RxN), dtype=float)
for datcut in range(0, MAXData):
    aMj = MC.Sour2aMj(datcut)  # 信源映射
    TxSig = MC.MIMO_MMCM(Cj, aMj) # 调制d
    for SampleCut in range(0, MAXTest_SamplePerData):  # 每个数据生成10万个测试样本
        label_Test[SampleCut + datcut * MAXTest_SamplePerData, datcut] = 1
        # MIMO信道这里是平坦衰落信道,后续添加多径多普勒衰落信道
        # H = np.random.rand(RxN, TxN) + 1j*np.random.rand(RxN, TxN)
        H = np.eye(RxN, TxN)
        noise = (np.random.rand(RxN, 1) + 1j*np.random.rand(RxN, 1))/50
        RxSig = np.dot(H, TxSig)  # 接收
        # 重组信号
        ReshapeRxSig = np.append(np.real(RxSig.reshape(-1)), np.imag(RxSig.reshape(-1)))
        image_Test[SampleCut + datcut * MAXTest_SamplePerData, :] = ReshapeRxSig

##########################################################
##########################################################
##########################################################
##########################################################
# 构建TensorFlow网络
# create data
x_data = np.float32(image_train)
y_data = np.float32(label_train)
# print(y_data)
xs = tf.placeholder(tf.float32, [None, 2*N*RxN])
ys = tf.placeholder(tf.float32, [None, MAXData])

# 第一层神经网络
W0 = tf.Variable(tf.truncated_normal([2*N*RxN, N*RxN]), 0.1)
b0 = tf.Variable(tf.zeros([N*RxN]))
y0 = tf.nn.sigmoid(tf.matmul(xs, W0) + b0)
# 第二层神经网络
W1 = tf.Variable(tf.truncated_normal([N*RxN, int(N*RxN/2)]), 0.1)
b1 = tf.Variable(tf.zeros([N*RxN/2]))
y1 = tf.nn.sigmoid(tf.matmul(y0, W1) + b1)
# 第三层神经网络
W2 = tf.Variable(tf.truncated_normal([int(N*RxN/2), int(N*RxN/2)]), 0.1)
b2 = tf.Variable(tf.zeros([N*RxN/2]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
# 第四层网络 softmax层
W3 = tf.Variable(tf.truncated_normal([int(N*RxN/2), MAXData]), 0.1)
b3 = tf.Variable(tf.zeros([MAXData]))
y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)
#####################################################
# 定义交叉熵
cross_entropy = -tf.reduce_sum(ys*tf.log(y3))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
#计算准确率
cor = tf.equal(tf.argmax(y3, 1), tf.argmax(ys, 1))
aur = tf.reduce_mean(tf.cast(cor, tf.float32))
# 开始运行模型
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1,10000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        prob_val = sess.run(aur, feed_dict={xs: x_data, ys: y_data})
        print("第{}次迭代，训练准确率为{}".format(i, prob_val))





