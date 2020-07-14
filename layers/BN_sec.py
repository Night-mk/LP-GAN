'''
BN_sec.py用于实现Batch Normalization
'''

import numpy as np
import types
from Module import Module
from Parameter import Parameter
import secure_protocols as secp

class BatchNorm_sec(Module):
    def __init__(self, in_channels, bit_length=32):
        super(BatchNorm_sec, self).__init__()
        # input_shape = [batchsize, channel_num, h, w] 
        self.in_channels = in_channels
        self.bit_length = bit_length

        # 初始化BN层需要的参数gamma、beta,参数的数量和均值、方差后的维度相同
        param_gamma = np.ones(self.in_channels)
        param_beta = np.zeros(self.in_channels)
        self.gamma = Parameter(param_gamma, requires_grad=True)
        self.beta = Parameter(param_beta, requires_grad=True)

        self.epsilon = 1e-5
        self.mean = np.zeros(self.in_channels)
        self.var = np.zeros(self.in_channels)

        # 初始化指数加权移动平均的mean和var，用于在测试环节计算样本均值和方差的无偏估计
        self.moving_mean = np.zeros(self.in_channels)
        self.moving_var = np.zeros(self.in_channels)
        self.moving_decay = 0.9

    def set_gamma(self, gamma):
        if isinstance(gamma, Parameter):
            self.gamma = gamma

    def set_beta(self, beta):
        if isinstance(beta, Parameter):
            self.beta = beta

    # 设置module打印格式
    def extra_repr(self):
        s = ('in_channels={in_channels}, eps={epsilon}, moving_decay={moving_decay}')
        return s.format(**self.__dict__)

    # BN前向传播，分两种模式：训练模式train，测试模式test
    # batchNormalizaion2d,均值和方差的均在channel_num维度求
    def forward(self, input_array_1, input_array_2, mode="train"):
        # input_data = [batch,channel_num,h,w]
        # self.input_data = input_array
        self.input_data_1 = input_array_1
        self.input_data_2 = input_array_2
        self.input_shape = self.input_data_1.shape
        self.batchsize = self.input_shape[0]

        # 计算均值的数据总量的维度m
        self.m = self.batchsize
        if self.input_data_1.ndim == 4:
            self.m = self.batchsize*self.input_data_1.shape[2]*self.input_data_1.shape[3]
        # 计算均值mean (axis=1对列求平均值,axis=0对行求平均)
        # keepdims=True可以保证使用np计算均值或者方差的结果保留原始数据的维度大小，可以方便的用于和原输入进行运算
        # self.mean = np.mean(self.input_data, axis=(0,2,3), keepdims=True)
        self.mean_1 = np.mean(self.input_data_1, axis=(0,2,3), keepdims=True)
        self.mean_2 = np.mean(self.input_data_2, axis=(0,2,3), keepdims=True)
        # print('mean.shape: ',self.mean.shape)
        # 记录一下标准差standard矩阵, 反向传播时使用
        # self.standard = self.input_data-self.mean
        self.standard_1 = self.input_data_1-self.mean_1
        self.standard_2 = self.input_data_2-self.mean_2
        # 计算方差var
        ## 计算标准差的平方
        # print('bn mul_1 shape: ', self.standard_1.shape)
        std_sq_1, std_sq_2 = secp.SecMul_matrix(self.standard_1, self.standard_1, self.standard_2, self.standard_2, bit_length=self.bit_length)
        self.var_1 = np.mean(std_sq_1, axis=(0,2,3), keepdims=True)
        self.var_2 = np.mean(std_sq_2, axis=(0,2,3), keepdims=True)
        # self.var = np.var(self.input_data_1+self.input_data_2, axis=(0,2,3), keepdims=True)
        # 存在多组数据batch的情况下，需要计算方差的无偏估计（b/(b-1)*E(var(x))） [但是pytorch似乎也没这么计算]
        # if self.batchsize>1:
        #     self.var = self.m/(self.m-1)*self.var
        
        # 利用指数加权平均算法计算moving_mean和moving_var,用于测试时作为整体的mean,var的无偏估计
        if np.sum(self.moving_mean)==0 and np.sum(self.moving_var)==0:
            self.moving_mean = self.mean_1+self.mean_2
            self.moving_var = self.var_1+self.var_2
        else:
            self.moving_mean = self.moving_decay * self.moving_mean + (1-self.moving_decay)*(self.mean_1+self.mean_2)
            self.moving_var = self.moving_decay * self.moving_var + (1-self.moving_decay)*(self.var_1+self.var_2)

        # 计算标准化值normed_x = [batch, bn_shape]
        if mode=='train':
            tua_sqrt = 5
            # print('sqrt input_sec: ', self.var_1[0][0]+self.epsilon, self.var_2[0][0])
            # print('sqrt shape: ', self.var_1.shape)
            # self.normed_x = (self.input_data-self.mean)/np.sqrt(self.var+self.epsilon)
            # print('bn sqrt shape: ', self.var_1.shape)
            inverse_sq_x_1, inverse_sq_x_2 = secp.SSqrt(self.var_1+self.epsilon, self.var_2, tua_sqrt, inverse_required=True, bit_length=self.bit_length)
            # print('error var: ',self.var-(self.var_1+self.var_2))

            # print('error inverse sqrt: ',(inverse_sq_x_1+inverse_sq_x_2 - 1/np.sqrt(self.var_1+self.var_2+self.epsilon))[0][0])
            # print('bn mul_1 shape: ', self.standard_1.shape)
            self.normed_x_1, self.normed_x_2 = secp.SecMul_matrix(self.standard_1, inverse_sq_x_1, self.standard_2, inverse_sq_x_2)
            # self.normed_x = (self.standard_1+self.standard_2)/np.sqrt(self.var+self.epsilon)
            # print('error inverse normed_x: ',self.normed_x-(self.normed_x_1+self.normed_x_2))
            # print(self.normed_x)
        '''test的先不写了'''
        if mode=='test':
            self.normed_x = (self.input_data-self.moving_mean)/np.sqrt(self.moving_var+self.epsilon)
        # 计算BN输出 output_y = [batch, -1]
        # 对每个输入都进行标准化，所以输出y的size和输入相同
        # print('gamma.shape: ',self.gamma.shape)
        # print('normed_x.shape: ',self.normed_x.shape)
        # print('type_gamma: ',self.gamma[0])
        # print('type_normed_x: ',type(self.normed_x))
        
        # 对每个channel做一次线性变换
        output_y_1 = np.zeros(self.input_shape)
        output_y_2 = np.zeros(self.input_shape)
        # output_y = np.zeros(self.input_shape)
        for i in range(self.in_channels):
            # output_y[:,i,:,:] = self.gamma.data[i]*self.normed_x[:,i,:,:] + self.beta.data[i]
            # output_y[:,i,:,:] = output_y_i
            output_y_1[:,i,:,:] = self.gamma.data[i]*self.normed_x_1[:,i,:,:] + self.beta.data[i]
            output_y_2[:,i,:,:] = self.gamma.data[i]*self.normed_x_2[:,i,:,:]
        # output_y = np.array(output_y)
        # output_y = self.gamma*self.normed_x + self.beta
        # print('error scale: ',output_y-(output_y_1+output_y_2))

        return output_y_1, output_y_2
    
    # 梯度计算函数
    def gradient(self, eta):
        # eta = [batch, channel_num, height, width]
        # 无论上层误差如何，首先将上层传输的误差转化为[batch, -1]
        self.eta = eta
        # self.gamma.grad = np.sum(self.eta*self.normed_x, axis=(0,2,3), keepdims=True)
        self.gamma.grad = np.sum(self.eta*self.normed_x, axis=(0,2,3))
        # self.beta.grad = np.sum(self.eta, axis=(0,2,3), keepdims=True)
        self.beta.grad = np.sum(self.eta, axis=(0,2,3))
        # 计算向前一层传播的误差参数
        # normed_x_grad
        # normed_x_grad = self.eta*self.gamma 
        # 由于eta[B,C,H,W] gamma=[C],直接乘维度不对，需要针对C这个维度进行乘法，最后还原输出的size[B,C,H,W]
        normed_x_grad = np.zeros(self.eta.shape)
        for i in range(self.in_channels):
            normed_x_grad_i = self.gamma.data[i]*self.eta[:,i,:,:]
            normed_x_grad[:,i,:,:] = normed_x_grad_i
        # print(self.eta)
        # var_grad
        var_grad = -1.0/2*np.sum(normed_x_grad*self.standard, axis=(0,2,3), keepdims=True)/(self.var+self.epsilon)**(3.0/2)
        # mean_grad
        mean_grad = -1*np.sum(normed_x_grad/np.sqrt(self.var+self.epsilon), axis=(0,2,3), keepdims=True) + var_grad*np.sum(-2*self.standard,axis=(0,2,3), keepdims=True)/self.m
        # input_grad
        input_grad = normed_x_grad/np.sqrt(self.var+self.epsilon) + var_grad*2*self.standard/self.m + mean_grad/self.m

        self.eta_next = input_grad
        return input_grad

def bn_test():
    # shape = np.array([12,3])
    # x = np.arange(36).reshape(shape)

    a = np.arange(48).reshape((4,3,2,2))
    bn1 = BatchNorm(3)
    bn_out = bn1.forward(a, 'train')
    # print(a)
    # print(bn1.input_data)
    # print(bn1.mean)
    # print(bn1.var)
    # print(bn1.normed_x)
    print('bn_out: ',bn_out)
    print('bn_out: ',bn_out.shape)

    dy=np.array([[[[1.3028, 0.5017],
       [-0.8432, -0.2807]],

      [[-0.4656, 0.2773],
       [-0.7269, 0.1338]]],

     [[[-3.1020, -0.7206],
       [0.4891, 0.2446]],

      [[0.2814, 2.2664],
       [0.8446, -1.1267]]],

     [[[-2.4999, 1.0087],
       [0.6242, 0.4253]],

      [[2.5916, 0.0530],
       [0.5305, -2.0655]]]])

    # x_grad = bn1.gradient(dy)
    # print(x_grad[0,0])
    

if __name__ == '__main__':
    # basic_test()
    bn_test()