'''
    Deconv.py 用于实现转置卷积（反卷积）计算，可以使用在GAN网络中
'''
import numpy as np
import Conv
from Module import Module
from Parameter import Parameter
import secure_protocols as secp

"""根据stride填充矩阵"""
def padding_stride(input_array, stride):
    pad_size = (input_array.shape[3]-1)*(stride-1)+input_array.shape[3]
    input_array_pad = np.zeros((input_array.shape[0], input_array.shape[1], pad_size, pad_size))
    for i in range(0, input_array.shape[3]):
        for j in range(0, input_array.shape[3]):
            input_array_pad[:,:, stride*i, stride*j] = input_array[:,:, i, j]
    return input_array_pad

"""额外填充矩阵（左边，上边各填充一行0）"""
def padding_additional(input_array):
    input_array_pad = np.pad(input_array, ((0,0), (0,0), (1,0), (1,0)), 'constant', constant_values=0)
    return input_array_pad

"""转置卷积类Deconv"""
class Deconv_sec(Module):
    def __init__(self, in_channels, out_channels, filter_size, zero_padding, stride, method='SAME', bias_required=True, bit_length=32):
        super(Deconv_sec, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.zero_padding = zero_padding
        self.stride = stride
        self.method = method
        self.bias_required = bias_required
        self.bit_length = bit_length
        
        # 定义参数
        param_weights_1 = np.random.uniform(-1e-2, 1e-2, (self.out_channels, self.in_channels, self.filter_size, self.filter_size))
        param_weights_2 = np.random.uniform(-1e-2, 1e-2, (self.out_channels, self.in_channels, self.filter_size, self.filter_size))
        self.weights_1 = Parameter(param_weights_1, requires_grad=True)
        self.weights_2 = Parameter(param_weights_2, requires_grad=True)
        if self.bias_required:
            param_bias_1 = np.zeros(self.out_channels)
            param_bias_2 = np.zeros(self.out_channels)
            self.bias_1 = Parameter(param_bias_1, requires_grad=True)
            self.bias_2 = Parameter(param_bias_2, requires_grad=True)
        else:
            self.bias_1 = None
            self.bias_2 = None

    # 静态方法计算反卷积层输出尺寸大小 
    # O=(W-1)*S-2P+F [(O-F+2P)%S==0]
    # O=(W-1)*S-2P+F+(O-F+2P)%S [(O-F+2P)%S!=0]
    @staticmethod
    def compute_output_size(input_size, filter_size, zero_padding, stride):
        output_size = (input_size-1)*stride-2*zero_padding+filter_size
        residue = (output_size-filter_size+2*zero_padding)%stride

        if residue==0: return output_size
        else: return output_size+residue

    # 设置特定的权重和偏移量
    def set_weight_1(self, weight):
        if isinstance(weight, Parameter):
            self.weights_1 = weight

    def set_bias_1(self, bias):
        if isinstance(bias, Parameter) and self.bias_required:
            self.bias_1 = bias

    # 设置特定的权重和偏移量
    def set_weight_2(self, weight):
        if isinstance(weight, Parameter):
            self.weights_2 = weight

    def set_bias_2(self, bias):
        if isinstance(bias, Parameter) and self.bias_required:
            self.bias_2 = bias

    # 设置module打印格式
    def extra_repr(self):
        s = ('in_channels={in_channels}, out_channels={out_channels}, kernel_size={filter_size}'
             ', stride={stride}, padding={zero_padding}')
        if self.bias_1 is None:
            s += ', bias=False'
        if self.method != None:
            s += ', method={method}'
        return s.format(**self.__dict__)

    # 前向传播计算
    # 需要填充输入矩阵，计算填充大小，并执行卷积计算
    def forward(self, input_array_1, input_array_2):
        self.input_array_1 = input_array_1
        self.input_array_2 = input_array_2
        self.input_shape = self.input_array_1.shape # [B,C,H,W]
        self.batchsize = self.input_shape[0]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        # 计算反卷积输出大小
        self.output_size = Deconv_sec.compute_output_size(self.input_width, self.filter_size, self.zero_padding, self.stride)

        self.output_array = np.zeros((self.batchsize, self.out_channels, self.output_size, self.output_size))  

        '''反卷积基础参数填充：需要对input进行两次填充'''
        # 第一次，根据stride在input内部填充0，每个元素间填充0的个数为n=stride-1
        input_pad_1 = self.input_array_1
        input_pad_2 = self.input_array_2
        if self.stride>=2:
            # input_pad = padding_stride(input_pad, self.stride)
            input_pad_1 = padding_stride(input_pad_1, self.stride)
            input_pad_2 = padding_stride(input_pad_2, self.stride)
        # print('input_pad first: ',(input_pad_1+input_pad_2)[0])
        # print('first pad: ', input_pad.shape)
        # 第二次填充，根据输出大小，计算以stride=1的卷积计算的输入需要的padding，如果padding%2不为0，则优先在input的左侧和上侧填充0【2P=(O-1)*1+F-W】
        # input_pad = Conv.padding(input_pad, self.method, self.zero_padding) # 必要填充
        input_pad_1 = Conv.padding(input_pad_1, self.method, self.zero_padding) # 必要填充
        input_pad_2 = Conv.padding(input_pad_2, self.method, self.zero_padding) # 必要填充
        # print('second pad: ', input_pad.shape)
        # print('second pad: ', (input_pad_1+input_pad_2)[0])

        '''反卷积中的卷积计算填充：需要对input再次填充'''
        padding_num_2 = (self.output_size-1+self.filter_size-input_pad_1.shape[3])
        # input_pad = Conv.padding(input_pad, 'SAME', padding_num_2//2) # 必要填充
        input_pad_1 = Conv.padding(input_pad_1, 'SAME', padding_num_2//2) # 必要填充
        input_pad_2 = Conv.padding(input_pad_2, 'SAME', padding_num_2//2) # 必要填充

        if padding_num_2%2!=0: # 在input的左侧和上侧填充0
            # input_pad = padding_additional(input_pad)
            input_pad_1 = padding_additional(input_pad_1)
            input_pad_2 = padding_additional(input_pad_2)
        # print('padding_num_2: ', padding_num_2)
        # print('third pad: ', input_pad.shape)
        # print('third pad: ', (input_pad_1+input_pad_2)[0])

        '''转换filter为矩阵'''
        ## 计算旋转180度的权重矩阵，rot180(W)
        # flip_weights = self.weights.data[...,::-1,::-1]
        # weights_col = flip_weights.reshape([self.out_channels, -1])
        # if self.bias_required:
        #     bias_col = self.bias.data.reshape([self.out_channels, -1])
        flip_weights_1 = self.weights_1.data[...,::-1,::-1]
        flip_weights_2 = self.weights_2.data[...,::-1,::-1]
        weights_col_1 = flip_weights_1.reshape([self.out_channels, -1])
        weights_col_2 = flip_weights_2.reshape([self.out_channels, -1])
        if self.bias_required:
            bias_col_1 = self.bias_1.data.reshape([self.out_channels, -1])
            bias_col_2 = self.bias_2.data.reshape([self.out_channels, -1])
        # print('weight_sec: ', (weights_col_1+weights_col_1))

        '''计算反卷积前向传播'''
        self.input_col_1 = []
        self.input_col_2 = []
        deconv_out_1 = np.zeros(self.output_array.shape)
        deconv_out_2 = np.zeros(self.output_array.shape)
        for i in range(0, self.batchsize):
            # input_i = input_pad[i][np.newaxis,:] #获取每个batch的输入内容
            input_i_1 = input_pad_1[i][np.newaxis,:]
            input_i_2 = input_pad_2[i][np.newaxis,:] 
            # input_col_i = Conv.img2col(input_i, self.filter_size, 1, self.zero_padding) #将每个batch的输入拉为矩阵(注意此处的stride=1)
            input_col_i_1 = Conv.img2col(input_i_1, self.filter_size, 1, self.zero_padding)
            input_col_i_2 = Conv.img2col(input_i_2, self.filter_size, 1, self.zero_padding)
            # print('Deconv input_i.shape: \n',input_i.shape)
            # print('Deconv input_col_i_1.shape: \n',input_col_i_1.shape)
            # print('Deconv weights_col_1.shape: \n',weights_col_1.shape)
            if self.bias_required:
                # deconv_out_i = np.dot(weights_col, input_col_i)+bias_col #计算矩阵卷积，输出大小为[Cout,(H-k+1)*(W-k+1)]的矩阵输出
                deconv_out_i_1, deconv_out_i_2 = secp.SecMul_dot_3(weights_col_1, input_col_i_1, weights_col_2, input_col_i_2)
                deconv_out_i_1 += bias_col_1
                deconv_out_i_2 += bias_col_2
            else:
                # deconv_out_i = np.dot(weights_col, input_col_i)
                deconv_out_i_1, deconv_out_i_2 = secp.SecMul_dot_3(weights_col_1, input_col_i_1, weights_col_2, input_col_i_2)
            # print('deconv_out_i.shape: \n',deconv_out_i.shape)
            # deconv_out[i] = np.reshape(deconv_out_i, self.output_array[0].shape) #转换为[Cout,Hout,Wout]的输出
            deconv_out_1[i] = np.reshape(deconv_out_i_1, self.output_array[0].shape)
            deconv_out_2[i] = np.reshape(deconv_out_i_2, self.output_array[0].shape) 

            # self.input_col_1.append(input_col_i_1) 
            # self.input_col_2.append(input_col_i_2) 
        # self.input_col_1 = np.array(self.input_col_1)
        # self.input_col_2 = np.array(self.input_col_2)
        # print('deconv out_shape: ',deconv_out_1.shape)
        # print('--------')

        return deconv_out_1, deconv_out_2

    # 计算w,b梯度，并计算向上一层传输的误差
    def gradient(self, eta):
        self.eta = eta # eta=[batch,out_c,out_h,out_w]
        # print('eta.shape: \n', eta.shape)
        eta_col = np.reshape(eta, [self.batchsize, self.out_channels, -1])
        # print('input_col.shape: \n', self.input_col.shape)
        # print('eta.shape: \n', self.eta.shape)
        
        '''计算weight，bias的梯度'''
        for i in range(0, self.batchsize):
            self.weights.grad += np.dot(eta_col[i], self.input_col[i].swapaxes(0,1)).reshape(self.weights.data.shape)[...,::-1,::-1]
        if self.bias_required:
            self.bias.grad += np.sum(eta_col, axis=(0, 2))
        # print('weight_grad.shape: \n', self.weights.grad.shape)

        '''计算向上一层传播的误差eta_next'''
        eta_pad = self.eta
        eta_pad = Conv.padding(eta_pad, self.method, self.zero_padding)
        # print('eta_pad.shape: \n', eta_pad.shape)

        flip_weights = self.weights.data.swapaxes(0, 1) # weights = [out_c, in_c, h, w]
        flip_weights_col = flip_weights.reshape([self.in_channels, -1])
        eta_next = []
        for i in range(0, self.batchsize):
            # print('eta_pad[i].shape: \n', eta_pad[i].shape)
            eta_pad_col_i = Conv.img2col(eta_pad[i][np.newaxis,:], self.filter_size, self.stride, self.zero_padding)
            # print('eta_pad_col_i.shape: \n', eta_pad_col_i.shape)
            eta_next_i = np.dot(flip_weights_col, eta_pad_col_i)
            eta_next.append(eta_next_i)
        self.eta_next = np.array(eta_next)
        self.eta_next = np.reshape(self.eta_next, self.input_shape)
        
        return self.eta_next

def deconv_forward_test():
    print('-------forward_test-------')
    # arange生成的是浮点数序列
    input_img = np.arange(48).reshape(1,3,4,4)
    # input_img = np.arange(192).reshape(1,3,8,8)
    de_cl1 = Deconv(in_channels=3, out_channels=3, filter_size=4,  zero_padding=1, stride=2, method='SAME')
    print('input_img', input_img)
    # forward
    deconv_out = de_cl1.forward(input_img)
    print(deconv_out)
    print('-----shape-----', deconv_out.shape)

    dy_numpy = np.random.random(deconv_out.shape).astype(np.float32)
    x_grad_numpy = de_cl1.gradient(dy_numpy)

    print('x_grad_numpy: \n', x_grad_numpy)
    print('-----shape-----', x_grad_numpy.shape)

def deconv_backward_test():
    return 0

if __name__ == "__main__":
    deconv_forward_test()