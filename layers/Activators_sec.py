'''
    Activators.py文件构建激活函数的前向，反向传播计算
'''
import numpy as np
from Module import Module
import secure_protocols as secp
import secure_protocols_2 as secp2
import secure_protocols_3 as secp3
import time

'''
    Square
'''
class Square(Module):
    def __init__(self, bit_length=32):
        self.bit_length = bit_length
        super(Square, self).__init__()
    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s
    # 前向传播
    def forward(self, input_array_1, input_array_2):
        # 求平方，做激活函数？？
        self.input_array_1 = input_array_1
        self.input_array_2 = input_array_2
        f1, f2 = secp.SSq(self.input_array_1, self.input_array_2, self.bit_length)
        # print('f1+f2: ', f1+f2)
        return f1, f2

    def gradient(self, eta_1, eta_2):
        self.eta_next_1 , self.eta_next_2 = secp.SecMul_matrix(eta_1, 2*self.input_array_1, eta_2, 2*self.input_array_2, self.bit_length)
        return self.eta_next_1, self.eta_next_2


'''
    ReLU
'''
class ReLU(Module):
    def __init__(self, bit_length=32):
        self.bit_length = bit_length
        super(ReLU, self).__init__()

    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s
    # 前向传播
    def forward(self, input_array_1, input_array_2):
        self.input_array_1 = input_array_1
        self.input_array_2 = input_array_2
        input_shape = self.input_array_1.shape
        input_zeros_1 = np.zeros(input_shape)
        input_zeros_2 = np.zeros(input_shape)
        # 使用0和input_array的元素依次比较
        # np.maximum：(X, Y, out=None) X与Y逐位比较取其大者
        index_matrix_z2_1, index_matrix_z2_2 ,self.offline_time, self.online_time, self.dfc_time = secp2.SecCmp_z2(self.input_array_1, input_zeros_1, self.input_array_2, input_zeros_2, self.bit_length) # 计时版
        # index_matrix_z2_1, index_matrix_z2_2 = secp3.SecCmp_z2(self.input_array_1, input_zeros_1, self.input_array_2, input_zeros_2, self.bit_length) # 非计时版
        # start_time = time.time()
        ## 将数域从Z2转到Z2**bit_length
        self.index_matrix_1, self.index_matrix_2 = secp2.SecFieldC(index_matrix_z2_1, index_matrix_z2_2, self.bit_length)
        # print("f1+f2: ",(self.f1+self.f2)&(2-1))
        # index_matrix用于保存input中的数是否大于0的矩阵
        # if u_i>0, index_matrix[i]=1
        # if u_i<=0, index_matrix[i]=0
        # self.index_matrix = (index_matrix_z2_1+index_matrix_z2_2)&(2-1) 
        # print('index_matrix_z2_1: \n', index_matrix_z2_1)
        # print('index_matrix_z2_2: \n', index_matrix_z2_2)
        # print('error: \n', self.index_matrix_1+self.index_matrix_2-self.index_matrix)
        relu_out_1, relu_out_2 = secp.SecMul_matrix(self.input_array_1, self.index_matrix_1, self.input_array_2, self.index_matrix_2, self.bit_length)
        # end_time = time.time()
        # print('SFC+Mulmatirx time: ',(end_time-start_time)*1000)
        # self.online_time += (end_time-start_time)*1000
        # print('total relu time: ', self.online_time+self.offline_time)
        # print('dfc_time: ', self.dfc_time*60*1000)
        # print('relu_shape: ', self.input_array_1.shape)
        # print('-----')
        
        return relu_out_1, relu_out_2, self.dfc_time
        # return self.index_matrix*self.input_array_1, self.index_matrix*self.input_array_2
    
    # 反向传播
    # relu'(x)=1*dy if x>0
    # relu'(x)=0*dy if x<0
    # relu'(x)不存在 if x=0.0000... （代码实现里将=0的结果设置为1）
    def gradient(self, eta_1, eta_2):
        # index_matrix = (self.f1+self.f2)&(2-1)
        # self.eta_next_1 = eta_1*self.index_matrix
        # self.eta_next_2 = eta_2*self.index_matrix
        self.eta_next_1, self.eta_next_2= secp.SecMul_matrix(eta_1, self.index_matrix_1, eta_2, self.index_matrix_2, self.bit_length)
        return self.eta_next_1, self.eta_next_2

'''
    LeakyReLU
'''
class LeakyReLU(Module):
    def __init__(self, alpha1=0.01, bit_length=32):
        # self.input_array = np.zeros(input_shape)
        # self.eta = np.zeros(input_shape)
        super(LeakyReLU, self).__init__
        self.alpha1 = alpha1
        self.bit_length = bit_length
    
    # 设置module打印格式
    def extra_repr(self):
        s = ('alpha={alpha1}')
        return s.format(**self.__dict__)

    # 前向传播
    # leakyrelu(x)=x if x>0
    # leakyrelu(x)=ax if x<=0
    # y(x)=c*x+(c xor 1)*alpha*x
    def forward(self, input_array_1, input_array_2):
        self.input_array_1 = input_array_1
        self.input_array_2 = input_array_2
        input_shape = self.input_array_1.shape
        input_zeros_1 = np.zeros(input_shape).astype(np.int)
        input_zeros_2 = np.zeros(input_shape).astype(np.int)
        input_ones = np.ones(input_shape).astype(np.int)
        '''计算C*x if x>0'''
        index_matrix_z2_1, index_matrix_z2_2 ,self.offline_time, self.online_time, self.dfc_time = secp2.SecCmp_z2(self.input_array_1, input_zeros_1, self.input_array_2, input_zeros_2, self.bit_length) # 计时版

        self.index_matrix_1, self.index_matrix_2 = secp2.SecFieldC(index_matrix_z2_1, index_matrix_z2_2, self.bit_length)

        lrelu_out_1, lrelu_out_2 = secp.SecMul_matrix(self.input_array_1, self.index_matrix_1, self.input_array_2, self.index_matrix_2, self.bit_length)
        '''计算alpha*x if x<=0'''
        # 对cmp输出与1做异或
        index_matrix_z2_alpha_1, index_matrix_z2_alpha_2, xor_online_time = secp2.SecXor_z2(index_matrix_z2_1, input_zeros_1, index_matrix_z2_2, input_ones)
        # 将异或结果转换域
        index_matrix_new_1, index_matrix_new_2 = secp2.SecFieldC(index_matrix_z2_alpha_1, index_matrix_z2_alpha_2, self.bit_length)
        lrelu_alpha_1, lrelu_alpha_2 = secp.SecMul_matrix(self.alpha1*self.input_array_1, index_matrix_new_1, self.alpha1*self.input_array_2, index_matrix_new_2, self.bit_length)

        lrelu_out_1+=lrelu_alpha_1
        lrelu_out_2+=lrelu_alpha_2

        return lrelu_out_1, lrelu_out_2, self.dfc_time
        # self.output_array = self.input_array.copy()
        # self.output_array[self.input_array<0] *= self.alpha1
        # return self.output_array

    # 反向传播
    def gradient(self, eta):
        self.eta_next = eta
        # print('eta shape: ',eta.shape)
        self.eta_next[self.input_array<=0] *= self.alpha1
        return self.eta_next

'''
    Sigmoid
'''
class Sigmoid_cmp(Module):
    def __init__(self, bit_length=32):
        super(Sigmoid_cmp, self).__init__()
        self.bit_length = bit_length
    
    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s
    # 1/(1+e^-x)
    def forward(self, input_array_1, input_array_2):
        eps=1e-5
        # CMP
        input_shape = input_array_1.shape
        a_1, a_2, offline_time_1, online_time_1 = secp2.SecCmp_z2(input_array_1, -0.5*np.ones(input_shape), input_array_2, np.zeros(input_shape), bit_length=self.bit_length) 
        b_1, b_2, offline_time_2, online_time_2 = secp2.SecCmp_z2(input_array_1, 0.5*np.ones(input_shape), input_array_2, np.zeros(input_shape), bit_length=self.bit_length)
        start_time = time.time()
        # XOR, AND
        alpha_1, alpha_2 = secp3.SecXor_z2(a_1,1,a_2,0)
        beta_x_1, beta_x_2 = secp3.SecXor_z2(b_1,1,b_2,0)
        beta_1, beta_2 = secp3.SecMul_z2(a_1, beta_x_1, a_2, beta_x_2)
        # FIELD CONVERT
        alpha_1, alpha_2 = secp2.SecFieldC(alpha_1, alpha_2, bit_length=self.bit_length)
        beta_1, beta_2 = secp2.SecFieldC(beta_1, beta_2, bit_length=self.bit_length)
        gamma_1, gamma_2 = secp2.SecFieldC(b_1, b_2, bit_length=self.bit_length)
        # MUL
        alpha_1, alpha_2 = secp.SecMul_matrix(alpha_1, eps, alpha_2, 0, bit_length=self.bit_length)
        beta_1, beta_2 = secp.SecMul_matrix(beta_1, input_array_1+0.5, beta_2, input_array_2, bit_length=self.bit_length)
        gamma_1, gamma_2 = secp.SecMul_matrix(gamma_1, 1-eps, gamma_2, 0, bit_length=self.bit_length)
        self.output_array_1 = alpha_1+beta_1+gamma_1
        self.output_array_2 = alpha_2+beta_2+gamma_2
        end_time = time.time()
        self.offline_time = offline_time_1+offline_time_2
        self.online_time = online_time_1+online_time_2+(end_time-start_time)*1000
        return self.output_array_1, self.output_array_2
        # self.output_array = 1/(1+np.exp(-input_array))
        # return self.output_array
    
    def gradient(self, eta):
        self.eta_next = eta * self.output_array*(self.output_array-1) # d(sigmoid)=y*(1-y)
        return self.eta_next

'''DCGAN中使用的Sigmoid'''
class Sigmoid_CE_sec(Module):
    def __init__(self, bit_length=32):
        super(Sigmoid_CE_sec, self).__init__()
        self.bit_length = bit_length
    
    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s
    # 1/(1+e^-x)
    def forward(self, input_array_1, input_array_2):
        input_array = input_array_1+input_array_2
        input_array_1 = input_array/3
        input_array_2 = input_array-input_array_1
        tau = 10
        initial_num = 1e-2 # 训练的时候调整一下
        t1 = np.exp(-input_array_1)
        t2 = np.exp(-input_array_2) 
        u1, u2 = secp.SecMul_matrix(t1, t2 ,0, 0, self.bit_length)
        self.output_array_1, self.output_array_2 = secp.SecInv(1+u1, u2, tau, initial=initial_num, bit_length=self.bit_length) 
        # print('1+exp(-1):', 1+np.exp(-(input_array_1+input_array_2)))    
        # print('1+exp(-1)_sec:', 1+u1+u2)    
        return self.output_array_1, self.output_array_2   
        # self.output_array = 1/(1+np.exp(-input_array))
        # return self.output_array
    
    def gradient(self, eta_1, eta_2):
        n_dim = self.output_array.ndim
        for i in range(n_dim-1):
            eta = eta[:,np.newaxis]
        # eta = y, grad = sigmoid(x)-y
        self.eta_next = self.output_array - eta
        self.eta_next /= self.output_array.shape[0] # 需要求平均
        # print('sigmoid eta shape: \n', eta.shape)
        # print('sigmoid output_array shape: \n', self.output_array.shape)
        return self.eta_next
    '''
    def gradient(self, eta):
        n_dim = self.output_array.ndim
        for i in range(n_dim-1):
            eta = eta[:,np.newaxis]
        # eta = y, grad = sigmoid(x)-y
        self.eta_next = self.output_array - eta
        self.eta_next /= self.output_array.shape[0] # 需要求平均
        # print('sigmoid eta shape: \n', eta.shape)
        # print('sigmoid output_array shape: \n', self.output_array.shape)
        return self.eta_next
    '''
'''
    tanh
'''
class Tanh_sec(Module):
    def __init__(self,bit_length):
        super(Tanh_sec, self).__init__()
        self.bit_length = bit_length
    
    # 设置module打印格式
    def extra_repr(self):
        s = ()
        return s

    # tanh=2*sigmoid(2x)-1
    def forward(self, input_array_1, input_array_2):
        # self.output_array = 2/(1+np.exp(-2*input_array))-1
        # return self.output_array
        # shares再分配= =策略
        input_array = input_array_1+input_array_2
        input_array_1 = input_array/3
        input_array_2 = input_array-input_array_1
        tau = 10
        initial_num = 1e-2
        t1 = np.exp(-2*input_array_1)
        t2 = np.exp(-2*input_array_2) 
        u1, u2 = secp.SecMul_matrix(t1, t2, 0, 0, self.bit_length)
        output_array_1, output_array_2 = secp.SecInv(1+u1, u2, tau, initial=initial_num, bit_length=self.bit_length) 
        self.output_array_1 = 2*output_array_1-1
        self.output_array_2 = 2*output_array_2

        return self.output_array_1, self.output_array_2

    def gradient(self, eta_1, eta_2):
        # self.eta_next = eta * (1-self.output_array**2)
        # return self.eta_next
        sq_1, sq_2 = secp.SecMul_matrix(self.output_array_1, self.output_array_1, self.output_array_2, self.output_array_2, self.bit_length)
        self.eta_next_1, self.eta_next_2 = secp.SecMul_matrix(eta_1, 1-sq_1, eta_2, -sq_2, self.bit_length)
        return self.eta_next_1, self.eta_next_2
        

def test_leakyrelu():
    x = np.random.randn(1,1,4,4).astype(np.float32)
    dy = np.random.randn(1,1,4,4).astype(np.float32)
    print('x: \n', x)
    print('dy: \n', dy)

    lrelu = LeakyReLU()
    l_out = lrelu.forward(x)
    l_eta = lrelu.gradient(dy)
    print(l_out)
    print('----------')
    print(l_eta)

def test_relu():
    x = np.random.randn(1,1,4,4).astype(np.float32)
    dy = np.random.randn(1,1,4,4).astype(np.float32)
    print('x: \n', x)
    print('dy: \n', dy)

    relu = ReLU()
    l_out = relu.forward(x)
    l_eta = relu.gradient(dy)
    
    print(l_out)
    print('----------')
    print(l_eta)

if __name__ == "__main__":
    test_leakyrelu()
    # test_relu()