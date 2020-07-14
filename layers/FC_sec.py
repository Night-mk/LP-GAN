'''
FC.py用于实现Fully Connected Layer
全连接层应该相当于DNN吧，深度神经网络，第l层每个神经元和l+1层每个神经元都相连
'''

import numpy as np
from Module import Module
from Parameter import Parameter
import secure_protocols as secp

class FullyConnect_sec(Module):
    def __init__(self, in_num, out_num, bias_required=True, bit_length=32):
        super(FullyConnect_sec, self).__init__()
        # input_shape = [batchsize, channel_num, height, width](卷积层)
        #  or [batchsize, input_num](全连接层)
        self.in_num = in_num
        # output_shape = [batchsize, out_num] 其实单个output就是个一维数组,列向量
        self.out_num = out_num
        self.bias_required = bias_required
        self.bit_length = bit_length
        '''使用xavier初始化'''
        # 初始化全连接层为输入的weights
        # param_weights = np.random.standard_normal((self.in_num, self.out_num))/100
        param_weights = self.xavier_init(self.in_num, self.out_num, (self.in_num, self.out_num))
        param_weights_1 = self.xavier_init(self.in_num, self.out_num, (self.in_num, self.out_num))
        param_weights_2 = param_weights-param_weights_1
        # self.weights = Parameter(param_weights, requires_grad=True)
        self.weights_1 = Parameter(param_weights_1, requires_grad=True)
        self.weights_2 = Parameter(param_weights_2, requires_grad=True)
        # bias初始化为列向量
        # param_bias = np.random.standard_normal(self.out_num)/100
        if self.bias_required:
            param_bias = self.xavier_init(self.in_num, self.out_num, (self.out_num))
            param_bias_1 = self.xavier_init(self.in_num, self.out_num, (self.out_num))
            param_bias_2 = param_bias - param_bias_1
            # self.bias = Parameter(param_bias, requires_grad=True)
            self.bias_1 = Parameter(param_bias_1, requires_grad=True)
            self.bias_2 = Parameter(param_bias_2, requires_grad=True)
        else:
            # self.bias = None
            self.bias_1 = None
            self.bias_2 = None

    # 设置特定的权重和偏移量
    def set_weight_1(self, weight):
        if isinstance(weight, Parameter):
            self.weights_1 = weight

    def set_weight_2(self, weight):
        if isinstance(weight, Parameter):
            self.weights_2 = weight

    def set_bias_1(self, bias):
        if isinstance(bias, Parameter) and self.bias_required:
            self.bias_1 = bias

    def set_bias_2(self, bias):
        if isinstance(bias, Parameter) and self.bias_required:
            self.bias_2 = bias

    def xavier_init(self, fan_in, fan_out, shape, constant=1):
        # 这个初始化对不收敛的问题没啥帮助= =
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low, high, shape)

    # 设置module打印格式
    def extra_repr(self):
        s = ('in_features={in_num}, out_features={out_num}')
        if self.bias_1 is None:
            s += ', bias=False'
        if self.bias_required:
            s += ', bias=True'
        return s.format(**self.__dict__)

    # 前向传播计算
    def forward(self, input_array_1, input_array_2):
        self.input_shape = input_array_1.shape
        self.batchsize = self.input_shape[0]
        # 对batchsize中的每个输入数据进行全连接计算
        # input_col=[batchsize, in_num], weights=[in_num, out_num]
        # 这种结构适合使用batch计算
        self.input_col_1 = input_array_1.reshape([self.batchsize, -1])
        self.input_col_2 = input_array_2.reshape([self.batchsize, -1])
        # print('input_shape: \n', self.input_col.shape)
        '''
            [Z1,Z2,...Zm]=[m x n矩阵]*[A1,A2,...An]+[B1,B2,...Bm]
            输入输出均拉为列向量
        '''
        # output_array = [batchsize, out_num]
        # output_array = np.dot(self.input_col, self.weights.data) + self.bias.data
        output_array_1, output_array_2 = secp.SecMul_dot_3(self.input_col_1, self.weights_1.data, self.input_col_2, self.weights_2.data, self.bit_length)
        output_array_1 += self.bias_1.data
        output_array_2 += self.bias_2.data
        # print('output_shape: \n',output_array.shape)
        return output_array_1, output_array_2
            
    # 梯度计算函数
    def gradient(self, eta_1, eta_2):
        # eta=[batchsize, out_num]
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        bias_shape = self.bias_1.data.shape
        # print('eta.shape: \n',self.eta.shape)
        # DNN反向传播, 计算delta_W
        for i in range(0, self.eta_1.shape[0]):
            # input_col_i = self.input_col[i][:, np.newaxis]
            input_col_i_1 = self.input_col_1[i][:, np.newaxis]
            input_col_i_2 = self.input_col_2[i][:, np.newaxis]

            # eta_i = self.eta[i][:, np.newaxis].T
            eta_i_1 = self.eta_1[i][:, np.newaxis].T
            eta_i_2 = self.eta_2[i][:, np.newaxis].T
            # 利用每个batch输出参数误差累加计算梯度
            # weights=[out_num, in_num]
            # self.weights.grad += np.dot(input_col_i, eta_i)
            grad_i_1, grad_i_2 = secp.SecMul_dot_3(input_col_i_1, eta_i_1, input_col_i_2, eta_i_2, self.bit_length)
            self.weights_1.grad += grad_i_1
            self.weights_2.grad += grad_i_2
            # self.bias.grad += eta_i.reshape(self.bias.data.shape)
            self.bias_1.grad += eta_i_1.reshape(bias_shape)
            self.bias_2.grad += eta_i_2.reshape(bias_shape)

        # print('eta shape: \n',self.eta.shape)
        # print('weight.data shape: \n',self.weights.data.shape)
        # 计算上一层的误差 eta=[batch,out_num], weights=[in_num,out_num]
        # self.eta_next = np.dot(self.eta, self.weights.data.T) # eta_next=[batch, in_num]
        self.eta_next_1, self.eta_next_2 = secp.SecMul_dot_3(self.eta_1, self.weights_1.data.T, self.eta_2, self.weights_2.data.T, self.bit_length) # eta_next=[batch, in_num]

        # return self.eta_next
        return self.eta_next_1, self.eta_next_2

def test_FC():
    fc_input = np.arange(54).reshape(2,3,3,3)
    # print('input: ', fc_input)
    fc1 = FullyConnect(27, 10)
    # print('fc.weight: ', fc1.weights)
    fc1_out = fc1.forward(fc_input)
    print('fc_out: \n', fc1_out)
    print(fc1_out.shape)

    
    # copy()就是深度拷贝
    fc1_next = fc1_out.copy()+1
    print('fc1_next: ',fc1_next-fc1_out)
    fc1_next1 = fc1.gradient(fc1_next-fc1_out)
    print('fc_next error: ', fc1_next1)
    print(fc1.weights_grad)
    print(fc1.bias_grad)
    # 反向传播
    fc1.backward()
    

if __name__ == '__main__':
    test_FC()