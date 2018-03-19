'''
    作者 肖绵合  2018/03/17
    这是MIMO-MMCM的Python库
    MIMO-MMCM的核心思想是利用Chirp信号的抗多普勒抗多径等复杂信道环境等特点，利用深度学习
算法构建解调模型。2018年以前的DeepLearning-based MIMO检测论文都是基于已知信道状态信息
用来训练深度学习模型。在扩频信号的时带宽积足够大的情况下可免于信道估计
    而DeepLearning-based MIMO-Chirp充分利用了Chirp信号特点，离线状态下不需要知道CSI
便可训练出理想的DL模型。
    这个库的DL MIMO-MMCM模型将信源归为整数型量，信源取值范围由QAM调制阶数QAM_M、MMCM子带
个数M、子带时带宽积P、信源组个数J、发射天线个数TxN、接收天线个数RxN共同决定，本程序中不使用
空时编码。
    此外，DL模型采用TensorFlow搭建
'''
import numpy as np
# SysPara [P,M,J,QAM_M,TxN,RxN]
global P,M,J,QAM_M,TxN,RxN
def InitMIMO_MMCM_PyLibSysPara(SysPara):
    global P,M,J,QAM_M,TxN,RxN
    P     = SysPara[0]
    M     = SysPara[1]
    J     = SysPara[2]
    QAM_M = SysPara[3]
    TxN   = SysPara[4]
    RxN   = SysPara[5]

def Sour2aMj(Sour):#Sour为整数，他的取值范围为2**(np.sqrt(QAM_M)*TxN*M*J)
    AntDat = Sour2AntInt(Sour)
    MJDat = AntInt2AntBandM(AntDat)
    QAMSourDat = AntBandM2AntBandMJ(MJDat)
    aMj = AntBandMJ2QAMMap(QAMSourDat)
    return aMj


def Sour2AntInt(Sour):# Sour是单独的一个数，它的取值范围为 0～2**(TxN * np.sqrt(QAM_M))-1的整数
    temp_Sour = Sour
    if Sour > 2**(np.sqrt(QAM_M)*TxN*M*J) :print('SysPara Error')
    AntDat = np.zeros((TxN, 1), dtype=int)
    for AntCut in range(0, TxN):
        AntDat[AntCut] = temp_Sour % (QAM_M*M*J) #每根天线上加载的整数型数据，后续还要对其进行子带分割，星座映射
        temp_Sour = temp_Sour / (QAM_M*M*J)
    return AntDat # 返回一个列向量，该向量意味着天线加载的整数型数据。

def AntInt2AntBandM(AntDat):
    MJDat = np.zeros((TxN, M), dtype=int)
    for AntCut in range(0, TxN):
        tempAntDat = AntDat[AntCut]
        for m in range(0,M):
            MJDat[AntCut,m] = tempAntDat % (QAM_M*J)
            tempAntDat = tempAntDat / (QAM_M*J)
    return MJDat #返回一个二维数组，每一行代表的是每根天线加载的未经映射和分组的频域整数

def AntBandM2AntBandMJ(MJDat):
    QAMSourDat = np.zeros((TxN, M, J), dtype=int)
    for AntCut in range(0, TxN):
        for m in range(0, M):
            tempAntDat = MJDat[AntCut, m]
            for j_cut in range(0, J):
                QAMSourDat[AntCut, m, j_cut] = tempAntDat % QAM_M
                tempAntDat = tempAntDat / QAM_M
    return QAMSourDat #返回一个3维数组，每一行代表的是每根天线加载的未经映射的频域整数


def AntBandMJ2QAMMap(QAMSourDat): #仅支持4QAM 16QAM
    AntMAPDat = np.zeros((TxN, M,J), dtype=complex)
    if QAM_M == 4:
        for datcut in range(0, TxN):
            for m in range(0, M):
                for j_cut in range(0, J):
                    AntMAPDat[datcut, m, j_cut] = QAM4MAP(QAMSourDat[datcut, m, j_cut])
    elif QAM_M == 16:
        for datcut in range(0, TxN):
            for m in range(0, M):
                for j_cut in range(0, J):
                    AntMAPDat[datcut, m, j_cut] = QAM16MAP(QAMSourDat[datcut, m, j_cut])
    else:
        print('ERROR')
    return AntMAPDat #返回一个3维数组，每一行代表的是对应天线加载的经映射的频域复矢量


def QAM4MAP(dat):#单个数 格雷码映射
    reDat = np.array([0], dtype=complex)
    if dat == 0:
        reDat = np.array([-1+1j], dtype=complex)
    elif dat == 1:
        reDat = np.array([-1-1j], dtype=complex)
    elif dat == 2:
        reDat = np.array([ 1+1j], dtype=complex)
    elif dat == 3:
        reDat = np.array([ 1-1j], dtype=complex)
    else:
        print('ERROR')
    return reDat
def QAM16MAP(dat):#单个数 格雷码映射
    reDat = np.array([0], dtype=complex)
    if dat == 0:
        reDat = np.array([-3+3j], dtype=complex)
    elif dat == 1:
        reDat = np.array([-3+1j], dtype=complex)
    elif dat == 2:
        reDat = np.array([-3-3j], dtype=complex)
    elif dat == 3:
        reDat = np.array([-3-1j], dtype=complex)
    elif dat == 4:
        reDat = np.array([-1+3j], dtype=complex)
    elif dat == 5:
        reDat = np.array([-1+1j], dtype=complex)
    elif dat == 6:
        reDat = np.array([-1-3j], dtype=complex)
    elif dat == 7:
        reDat = np.array([-1-1j], dtype=complex)
    elif dat == 8:
        reDat = np.array([3+3j], dtype=complex)
    elif dat == 9:
        reDat = np.array([3+1j], dtype=complex)
    elif dat == 10:
        reDat = np.array([3-3j], dtype=complex)
    elif dat == 11:
        reDat = np.array([3-1j], dtype=complex)
    elif dat == 12:
        reDat = np.array([1+3j], dtype=complex)
    elif dat == 13:
        reDat = np.array([1+1j], dtype=complex)
    elif dat == 14:
        reDat = np.array([1-3j], dtype=complex)
    elif dat == 15:
        reDat = np.array([1-1j], dtype=complex)
    else:
        print('ERROR')
    return reDat

###############################################################
###############################################################
##################      MMCM调制相关    ########################
##################      MMCM调制相关    ########################
###############################################################
###############################################################

def MMCM(Cj, aj):
    N = P*M
    s = np.zeros((N, 1), dtype=complex)
    for jj in range(0, J):
        ifft_dat = np.transpose([np.fft.ifft(aj[:, jj])])
        Cj_dat = Cj[jj, :, :]
        debug_temp = np.dot(Cj_dat, ifft_dat)
        s = np.add(s, debug_temp)
    return s

def Gen_Cj():# 注意，这里的Cj没有用上IFFT模块
    # Gen_Ssj生成扩频基，
    # 基于短时平移正交概念，以exp(1j * pi * P * ((n - 1) / N). ^ 2);为基准
    # 平移Tp = N / J个采样点生成一个短时平移正交波形。一个产生J组波形。
    N = P*M
    Tp = N/J # 整数型变量
    ssj = np.zeros((N, J),dtype=complex)
    for jj in range(0,J):
        for n in range(0,N):
            ssj[n,jj] = np.exp(1j*np.pi*P*((n+Tp*jj)/N)**2)
    # 生成STSO信号结束
    ENM = np.kron(np.ones((P, 1)), np.eye(M))
    Cj = np.zeros((J, N, M), dtype=complex)
    for jj in range(0, J):
        Cj[jj, :, :] = M * np.dot(np.diag(ssj[:, jj]), ENM)
    return Cj


def MIMO_MMCM(Cj, aMj):# aMj与AntMAPDat对接
    N = P*M
    TxSig = np.zeros((TxN,N),dtype=complex)
    # TxSig是各个天线发射信号的时域波形
    # 第一行数据代表第一根天线发射的时域波形，et...
    for TxNCut in range(0, TxN):
        aj = aMj[TxNCut, :, :]
        AntTxDatTemp = MMCM(Cj, aj)
        TxSig[TxNCut, :] = np.transpose(AntTxDatTemp)
    return TxSig

def Calc_pinvHc(Cj):
    Hc = Cj[0, :, :]
    for jj in range(1, J):
        Hc = np.append(Hc, Cj[jj, :, :])
    pinvHc = np.linalg.pinv(Hc)
    return pinvHc

def DeMMCM(pinvHc, r): # 仅仅单发单收成立
    aj_emst = np.dot(pinvHc, r)
    return aj_emst


def DeMIMO_MMCM(RxSig,pinvHc):
    aMj_emst = np.zeros((RxN,M),dtype=complex)
    for RxNCut in range(0,RxN):
        r = np.transpose([RxSig[RxNCut, :]])
        aj_emst = DeMMCM(pinvHc, r)
        aMj_emst[RxNCut,:] = aj_emst
    return aMj_emst





















