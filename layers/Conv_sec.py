'''
Conv_sec.py 用于实现CNN卷积层隐私计算，包括：
ConvLayer类，处理CNN前向传播，反向传播

'''
import numpy as np
from Module import Module
from Parameter import Parameter
import secure_protocols as secp
import time

'''
通用函数：img2col
将图像在卷积窗口中的数拉成一行,每行k^2列,总共(out_h*out_w)行
[B,Cin,H,W]->[B,Cin*k*k,(H-k+1)*(W-k+1)]
'''
def img2col(input_array, filter_size, stride=1, zp=0):
    # input_array 4d tensor [batch, channel, height, width]
    output_matrix=[]
    width = input_array.shape[3]
    height = input_array.shape[2]
    # range的上限应该是(input_size - filter_size + 1)
    for i in range(0, height-filter_size+1, stride):
        for j in range(0, width-filter_size+1, stride):
            input_col = input_array[:, :, i:i+filter_size, j:j+filter_size].reshape([-1])
            # print('inputcol: \n', input_col.shape)
            output_matrix.append(input_col)
    output_matrix = np.array(output_matrix).T
    # print('output_matrix:', output_matrix.shape)
    # output_shape = [B,Cin*k*k,(H-k+1)*(W-k+1)] stride默认为1
    # output_matrix 2d tensor [height, width]
    # 输出之前需要转置
    return output_matrix

'''
通用函数: padding
填充方法["VALID"截取, "SAME"填充]
'''
def padding(input_array, method, zp):
    # "VALID"不填充
    if method=='VALID':
        return input_array
    # "SAME"填充
    elif method=='SAME':
        # (before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值
        input_array = np.pad(input_array, ((0, 0), (0, 0), (zp, zp), (zp, zp)), 'constant', constant_values=0)
        return input_array

"""额外填充矩阵（左边，上边各填充一行0）"""
def padding_additional(input_array):
    input_array_pad = np.pad(input_array, ((0,0), (0,0), (1,0), (1,0)), 'constant', constant_values=0)
    return input_array_pad

'''
    ConvLayer类，实现卷积层以及前向传播函数，反向传播函数
'''
class Conv_sec(Module):
    # 初始化卷积层函数
    # 参数包括：输入数据大小[batch大小、通道数、输入高度、输入宽度]，滤波器宽度、滤波器高度、滤波器数目、补零数目、步长、学习速率、补零方法
    def __init__(self, in_channels, out_channels, filter_width, filter_height, zero_padding, stride, method='VALID', bias_required=True, bit_length=32):
        super(Conv_sec, self).__init__()
        # input_array 4d tensor [batch, channel, height, width]
        self.in_channels = in_channels
        # filter参数
        self.filter_width = filter_width  # 过滤器的宽度
        self.filter_height = filter_height  # 过滤器的高度
        self.out_channels = out_channels  # 过滤器组的数量（每组filter算一个）,输出通道数量
        self.zero_padding = zero_padding  # 补0圈数
        self.stride = stride # 步幅
        self.method = method
        self.bias_required = bias_required
        self.bit_length = bit_length

        # 卷积层过滤器初始化 （模型也是要拆成secret sharing）
        '''filter_num = output_channel,就是卷积输出feature map的通道数'''
        # 初始化S1，S2的两个weights
        param_weights = np.random.uniform(-1e-2, 1e-2,(self.out_channels, self.in_channels, self.filter_height, self.filter_width))
        param_weights_1 = np.random.uniform(-1e-2, 1e-2,(self.out_channels, self.in_channels, self.filter_height, self.filter_width))/2
        param_weights_2 = param_weights-param_weights_1
        # self.weights = Parameter(param_weights, requires_grad=True)
        self.weights_1 = Parameter(param_weights_1, requires_grad=True)
        self.weights_2 = Parameter(param_weights_2, requires_grad=True)
        if self.bias_required:
            param_bias = np.zeros(self.out_channels)
            param_bias_1 = np.random.randint(-2**(bit_length//4), 2**(bit_length//4), self.out_channels).astype(np.float64)
            param_bias_2 = param_bias-param_bias_1
            # self.bias = Parameter(param_bias, requires_grad=True)
            self.bias_1 = Parameter(param_bias_1, requires_grad=True)
            self.bias_2 = Parameter(param_bias_2, requires_grad=True)
        else:
            # self.bias = None
            self.bias_1 = None
            self.bias_2 = None

    # 设置特定的权重和偏移量
    # def set_weight(self, weight):
    #     if isinstance(weight, Parameter):
    #         self.weights = weight

    # def set_bias(self, bias):
    #     if isinstance(bias, Parameter) and self.bias_required:
    #         self.bias = bias

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
    
    # 设置module打印格式
    def extra_repr(self):
        s = ('in_channels={in_channels}, out_channels={out_channels}, kernel_size={filter_width}'
             ', stride={stride}, padding={zero_padding}')
        if self.bias_1 is None:
            s += ', bias=False'
        if self.method != None:
            s += ', method={method}'
        return s.format(**self.__dict__)

    # 静态方法计算卷积层输出尺寸大小=(W-F+2P)/S+1
    @staticmethod
    def compute_output_size(input_size, filter_size, zero_padding, stride):
        # 使用/会得到浮点数，使用//向下取整
        return (input_size-filter_size+2*zero_padding)//stride+1

    # 前向传播函数 img2col
    def forward(self, input_array_1, input_array_2):
        '''初始化输入、输出数据size'''
        self.input_shape = input_array_1.shape
        self.batchsize = self.input_shape[0]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        # 卷积层输出宽度计算 
        self.output_width = Conv_sec.compute_output_size(self.input_width, self.filter_width, self.zero_padding, self.stride)
        # 卷积层输出高度计算
        self.output_height = Conv_sec.compute_output_size(self.input_height, self.filter_height, self.zero_padding, self.stride)
        # 卷积层输出矩阵初始化 [batch, output_channel, height, width]
        self.output_array = np.zeros((self.batchsize ,self.out_channels, self.output_height, self.output_width))

        '''计算卷积'''
        # 转换filter为矩阵, 将每个filter拉为一列, filter [Cout,depth,height,width]
        weights_col_1 = self.weights_1.data.reshape([self.out_channels, -1])
        weights_col_2 = self.weights_2.data.reshape([self.out_channels, -1])
        
        if self.bias_required:
            bias_col_1 = self.bias_1.data.reshape([self.out_channels, -1])
            bias_col_2 = self.bias_2.data.reshape([self.out_channels, -1])

        # padding方法计算填充关系
        input_pad_1 = padding(input_array_1, self.method, self.zero_padding)
        input_pad_2 = padding(input_array_2, self.method, self.zero_padding)

        # self.input_col = []
        self.input_col_1 = []
        self.input_col_2 = []

        conv_out_1 = np.zeros(self.output_array.shape)
        conv_out_2 = np.zeros(self.output_array.shape)

        # print('output_shape: \n', conv_out.shape)
        
        # 对输入数据batch的每个图片、特征图进行卷积计算
        # start_time = time.time()
        '''卷积计算'''
        for i in range(0, self.batchsize):
            input_i_1 = input_pad_1[i][np.newaxis,:]
            input_i_2 = input_pad_2[i][np.newaxis,:] 

            input_col_i_1 = img2col(input_i_1, self.filter_width, self.stride, self.zero_padding) #将每个batch的输入拉为矩阵
            input_col_i_2 = img2col(input_i_2, self.filter_width, self.stride, self.zero_padding) 
            
            # print('weight_col shape: \n', weights_col_1.shape)
            # print('input_col_i shape: \n', input_col_i_1.shape)

            if self.bias_required:
                # conv_out_i = np.dot(weights_col, input_col_i)+bias_col #计算矩阵卷积，输出大小为[Cout,(H-k+1)*(W-k+1)]的矩阵输出
                # 隐私计算部分
                conv_out_i_1, conv_out_i_2 = secp.SecMul_dot_3(weights_col_1, input_col_i_1, weights_col_2, input_col_i_2, self.bit_length)
                conv_out_i_1+= bias_col_1
                conv_out_i_2+= bias_col_2
            else:
                # conv_out_i = np.dot(weights_col, input_col_i)
                ## 隐私计算部分
                conv_out_i_1, conv_out_i_2 = secp.SecMul_dot_3(weights_col_1, input_col_i_1, weights_col_2, input_col_i_2, self.bit_length)
            # conv_out[i] = np.reshape(conv_out_i, self.output_array[0].shape) #转换为[Cout,Hout,Wout]的输出
            conv_out_1[i] = np.reshape(conv_out_i_1, self.output_array[0].shape) 
            conv_out_2[i] = np.reshape(conv_out_i_2, self.output_array[0].shape)
        # end_time = time.time()
        # print('time consume: \n', (end_time-start_time)*1000)

        #     self.input_col_1.append(input_col_i_1)
        #     self.input_col_2.append(input_col_i_2)
        # self.input_col_1 = np.array(self.input_col_1)
        # self.input_col_2 = np.array(self.input_col_2)
        
        return conv_out_1, conv_out_2
        
    # 计算卷积梯度
    def gradient(self, eta_1, eta_2):
        # print('eta_shape: ',eta_1.shape)
        # print('output_shape: ',self.output_array.shape)
        # eta表示上层（l+1层）向下层（l层）传输的误差
        # 即Z_ij, eta=[batch,Cout,out_h,out_w]
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        # print('eta.shape: \n', self.eta.shape)
        # eta_col=[batch,Cout,out_h*out_w]
        eta_col_1 = np.reshape(eta_1, [self.batchsize, self.out_channels, -1])
        eta_col_2 = np.reshape(eta_1, [self.batchsize, self.out_channels, -1])
        
        '''计算W的梯度矩阵 delta_W=a^(l-1) conv delta_Z^l'''
        for i in range(0, self.batchsize):
            # self.weights.grad += np.dot(eta_col[i], self.input_col[i].T).reshape(self.weights.data.shape)
            w1_grad, w2_grad = secp.SecMul_dot_3(eta_col_1[i], self.input_col_1[i].T, eta_col_2[i], self.input_col_2[i].T, self.bit_length)
            self.weights_1.grad += w1_grad.reshape(self.weights_1.data.shape)
            self.weights_2.grad += w2_grad.reshape(self.weights_1.data.shape)
        '''计算b的梯度矩阵'''
        # print('eta_col: \n',eta_col)
        # print('eta.shape: \n',self.eta.shape)
        if self.bias_required:
            # self.bias.grad += np.sum(eta_col, axis=(0, 2))
            self.bias_1.grad += np.sum(eta_col_1, axis=(0, 2))
            self.bias_2.grad += np.sum(eta_col_2, axis=(0, 2))
        
        """计算传输到上一层的误差"""
        ## 针对stride>=2时对误差矩阵的填充，需要在每个误差数据中间填充(stride-1) ##
        eta_pad_1 = self.eta_1
        eta_pad_2 = self.eta_2
        # eta_pad = self.eta
        if self.stride>=2:
            # 计算中间填充后矩阵的size
            # pad_size = (self.eta.shape[3]-1)*(self.stride-1)+self.eta.shape[3]
            pad_size = (self.eta_1.shape[3]-1)*(self.stride-1)+self.eta_1.shape[3]
            # eta_pad = np.zeros((self.eta.shape[0], self.eta.shape[1], pad_size, pad_size))
            eta_pad_1 = np.zeros((self.eta_1.shape[0], self.eta_1.shape[1], pad_size, pad_size))
            eta_pad_2 = np.zeros((self.eta_2.shape[0], self.eta_2.shape[1], pad_size, pad_size))
            for i in range(0, self.eta_1.shape[3]):
                for j in range(0, self.eta_1.shape[3]):
                    # eta_pad[:,:,self.stride*i,self.stride*j] = self.eta[:,:,i,j]
                    eta_pad_1[:,:,self.stride*i,self.stride*j] = self.eta_1[:,:,i,j]
                    eta_pad_2[:,:,self.stride*i,self.stride*j] = self.eta_2[:,:,i,j]
        # 使用输出误差填充零 conv rot180[weights]
        # 计算填充后的误差delta_Z_pad,即使用0在eta_pad四周填充,'VALID'填充数量为ksize-1，'SAME'填充数量为ksize/2
        if self.method=='VALID':
            # eta_pad = np.pad(eta_pad, ((0,0),(0,0),(self.filter_height-1, self.filter_height-1),(self.filter_width-1, self.filter_width-1)),'constant',constant_values = (0,0))
            eta_pad_1 = np.pad(eta_pad_1, ((0,0),(0,0),(self.filter_height-1, self.filter_height-1),(self.filter_width-1, self.filter_width-1)),'constant',constant_values = (0,0))
            eta_pad_2 = np.pad(eta_pad_2, ((0,0),(0,0),(self.filter_height-1, self.filter_height-1),(self.filter_width-1, self.filter_width-1)),'constant',constant_values = (0,0))

        same_pad_height = (self.input_height-1+self.filter_height-eta_pad_1.shape[2])
        same_pad_width = (self.input_width-1+self.filter_width-eta_pad_1.shape[3])
        if self.method=='SAME':
            # eta_pad = np.pad(eta_pad, ((0,0),(0,0),(same_pad_height, same_pad_height),(same_pad_width, same_pad_width)),'constant',constant_values = (0,0))
            eta_pad_1 = np.pad(eta_pad_1, ((0,0),(0,0),(same_pad_height//2, same_pad_height//2),(same_pad_width//2, same_pad_width//2)),'constant',constant_values = (0,0))
            eta_pad_2 = np.pad(eta_pad_2, ((0,0),(0,0),(same_pad_height//2, same_pad_height//2),(same_pad_width//2, same_pad_width//2)),'constant',constant_values = (0,0))
        if same_pad_height%2!=0: # 在input的左侧和上侧填充0
            # eta_pad = padding_additional(eta_pad)
            eta_pad_1 = padding_additional(eta_pad_1)
            eta_pad_2 = padding_additional(eta_pad_2)

        ## 计算旋转180度的权重矩阵，rot180(W)
        ##  self.weight[Cout,depth,h,w]
        # flip_weights = self.weights.data[...,::-1,::-1]
        # flip_weights = flip_weights.swapaxes(0, 1)
        # flip_weights_col = flip_weights.reshape([self.in_channels, -1])
        flip_weights_1 = self.weights_1.data[...,::-1,::-1]
        flip_weights_2 = self.weights_2.data[...,::-1,::-1]
        flip_weights_1 = flip_weights_1.swapaxes(0, 1)
        flip_weights_2 = flip_weights_2.swapaxes(0, 1)
        flip_weights_col_1 = flip_weights_1.reshape([self.in_channels, -1])
        flip_weights_col_2 = flip_weights_2.reshape([self.in_channels, -1])

        ## 计算向上一层传播的误差eta_next,采用卷积乘计算
        # 原本，delta_Z^(l)=delta_Z^(l+1) conv rot180(W^(l))
        # eta_next = []
        eta_next_1 = []
        eta_next_2 = []
        for i in range(0, self.batchsize):
            # eta_pad_col_i = img2col(eta_pad[i][np.newaxis,:], self.filter_width, 1, self.zero_padding)
            eta_pad_col_i_1 = img2col(eta_pad_1[i][np.newaxis,:], self.filter_width, 1, self.zero_padding)
            eta_pad_col_i_2 = img2col(eta_pad_2[i][np.newaxis,:], self.filter_width, 1, self.zero_padding)
            # eta_next_i = np.dot(flip_weights_col, eta_pad_col_i)
            eta_next_i_1, eta_next_i_2 = secp.SecMul_dot_3(flip_weights_col_1, eta_pad_col_i_1, flip_weights_col_2, eta_pad_col_i_2, self.bit_length)
            # eta_next.append(eta_next_i)
            eta_next_1.append(eta_next_i_1)
            eta_next_2.append(eta_next_i_2)
        # self.eta_next = np.array(eta_next)
        self.eta_next_1 = np.array(eta_next_1).reshape(self.input_shape)
        self.eta_next_2 = np.array(eta_next_2).reshape(self.input_shape)
        # input_shape就是上一层的output_shape
        # self.eta_next = self.eta_next.reshape(self.input_shape)

        return self.eta_next_1, self.eta_next_2

def cnn_forward_test():
    print('-------forward_test-------')
    # arange生成的是浮点数序列
    input_img = np.arange(27).reshape(1,3,3,3)
    cl1 = ConvLayer(3, 5, 2,2, zero_padding=0, stride=1, method='VALID')
    print('input_img', input_img)
    # forward
    conv_out = cl1.forward(input_img)
    print(conv_out)
    print('-----shape-----', conv_out.shape)


def cnn_backward_test():
    input_img = np.arange(27).reshape(1,3,3,3) # input
    cl1 = ConvLayer(3, 5,2,2, zero_padding=0, stride=1, method='VALID') # convlayer
    conv_out = cl1.forward(input_img) # forward calculation

    # 假设误差为1
    conv_out1 = conv_out.copy()+1
    eta_next = cl1.gradient(conv_out1-conv_out) # gradient calculation

    print('eta_next: \n', eta_next)
    print('cl1.weight_grad: \n',cl1.weights.grad)
    print('cl1.bias_grad: \n',cl1.bias.grad)
    
def unit_test():
    batchsize = 2
    u = np.random.randn(batchsize,4,3)
    v = np.random.randn(batchsize,4,3)
    print('u: \n', u)
    print('v: \n', v)
    u1 = np.random.randn(batchsize,4,3)
    u2 = u-u1
    v1 = np.random.randn(batchsize,4,3)
    v2 = v-v1
    
    start_time = time.time()
    f1, f2 = secp.SecMul_matrix(u1, v1, u2, v2)
    end_time = time.time()
    print('f: ',f1+f2)
    print('u*v: ',u*v)
    print('err: ',(f1+f2)-u*v)
    
if __name__ == '__main__':
    unit_test()
    # cnn_forward_test()
    # cnn_backward_test()