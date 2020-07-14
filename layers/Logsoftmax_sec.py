'''
Logsoftmax.py用于实现Log版本的softmax
'''
import numpy as np
from Module import Module
import secure_protocols as secp

class Logsoftmax_sec(Module):
    def __init__(self, bit_length=32):
        super(Logsoftmax_sec, self).__init__()
        # input_shape=[batch, class_num]
        self.bit_length = bit_length

    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s

    # def cal_softmax(self, input_array): # 明文
    #     softmax = np.zeros(input_array.shape)
    #     # 对每个batch的数据求softmax
    #     exps_i = np.exp(input_array-np.max(input_array))
    #     softmax = exps_i/np.sum(exps_i)
    #     # softmax[batch, class_num]
    #     return softmax

    def cal_softmax(self, input_array_1, input_array_2): # 密文
        softmax_1 = np.zeros(input_array_1.shape)
        softmax_2 = np.zeros(input_array_2.shape)
        # 对每个batch的数据求softmax
        exp_i_1, exp_i_2 = secp.SecExp(input_array_1-np.max(input_array_1+input_array_2), input_array_2, self.bit_length)
        # print('exp_sec: ', exp_i_1+exp_i_2)
        sum_exp_1 = np.sum(exp_i_1)
        sum_exp_2 = np.sum(exp_i_2)
        # print('sum exp_sec: ', sum_exp_1+sum_exp_2)
        # exps_i = np.exp(input_array-np.max(input_array))
        tau_inv = 10
        initial_inv = 1e-2
        sum_exp_inv_1, sum_exp_inv_2 = secp.SecInv(sum_exp_1, sum_exp_2, tau_inv, initial=initial_inv, bit_length=self.bit_length)
        softmax_1, softmax_2 = secp.SecMul_matrix(exp_i_1, sum_exp_inv_1, exp_i_2, sum_exp_inv_2, self.bit_length)
        # print('inv_sec: ', sum_exp_inv_1+sum_exp_inv_2)
        # print('softmax_sec: ', softmax_1+softmax_2)
        # softmax = exps_i/np.sum(exps_i)
        # softmax[batch, class_num]
        return softmax_1, softmax_2

    def forward(self, input_array_1, input_array_2):
        self.input_shape = input_array_1.shape
        self.batchsize = self.input_shape[0]
        # self.eta = np.zeros(self.input_shape)
        # prediction 可以表示从FC层输出的数据 [batch, class_num] 或者 [batch, [p1,p2,...,pk]]
        self.logsoftmax_1 = np.zeros(self.input_shape)
        self.logsoftmax_2 = np.zeros(self.input_shape)
        self.softmax_1 = np.zeros(self.input_shape)
        self.softmax_2 = np.zeros(self.input_shape)
        # 对每个batch的数据求softmax
        for i in range(self.batchsize):
            self.softmax_1[i], self.softmax_2[i] = self.cal_softmax(input_array_1[i], input_array_2[i])
            self.logsoftmax_1[i], self.logsoftmax_2[i] = secp.SecLog(self.softmax_1[i], self.softmax_2[i], tau=10, bit_length=self.bit_length)
            # self.logsoftmax[i] = np.log(self.softmax[i])
        # softmax[batch, class_num]
        # return self.logsoftmax
        return self.logsoftmax_1, self.logsoftmax_2

    def gradient(self, eta_1, eta_2):
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.eta_next_1 = self.softmax_1.copy()
        self.eta_next_2 = self.softmax_2.copy()
        # print('softmax :\n', self.eta_next)
        # y 的标签是 One-hot 编码
        if self.eta_1.ndim>1: # one-hot
            for i in range(self.batchsize):
                self.eta_next_1[i] += self.eta_1[i]
                self.eta_next_2[i] += self.eta_2[i]
        elif self.eta_1.ndim==1: # 非one-hot
            for i in range(self.batchsize):
                self.eta_next_1[i, -(self.eta_1[i]+self.eta_2[i])] -= 1
        # eta[batchsize, class_num]
        # 需要除以batchsize用于平均该批次的影响
        self.eta_next_1 /= self.batchsize
        self.eta_next_2 /= self.batchsize
        return self.eta_next_1, self.eta_next_2
    
    '''
    def forward(self, input_array):
        self.input_shape = input_array.shape
        self.batchsize = self.input_shape[0]
        self.eta = np.zeros(self.input_shape)
        # prediction 可以表示从FC层输出的数据 [batch, class_num] 或者 [batch, [p1,p2,...,pk]]
        self.logsoftmax = np.zeros(input_array.shape)
        self.softmax = np.zeros(input_array.shape)
        # 对每个batch的数据求softmax
        for i in range(self.batchsize):
            self.softmax[i] = self.cal_softmax(input_array[i])
            self.logsoftmax[i] = np.log(self.softmax[i])
        # softmax[batch, class_num]
        return self.logsoftmax
    
    def gradient(self, eta):
        self.eta = eta
        self.eta_next = self.softmax.copy()
        # print('softmax :\n', self.eta_next)
        # y 的标签是 One-hot 编码
        if self.eta.ndim>1: # one-hot
            for i in range(self.batchsize):
                self.eta_next[i] += self.eta[i]
        elif self.eta.ndim==1: # 非one-hot
            for i in range(self.batchsize):
                self.eta_next[i, -self.eta[i]] -= 1
        # eta[batchsize, class_num]
        # 需要除以batchsize用于平均该批次的影响
        self.eta_next /= self.batchsize
        return self.eta_next
    '''