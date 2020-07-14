'''
    secure_component_test.py 使用pytorch构建网络及其组件，并验证numpy实现的组件的正确性
    -- 卷积计算Conv_sec
    -- 激活层AC
    -- 批量标准化层BN
    -- 全连接层FC
    -- 池化层Pooling
    -- 损失函数CrossEntropyLoss
    -- 反卷积层Deconv
'''
import torch 
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np
from Parameter import Parameter
from Conv_sec import Conv_sec
from Conv import ConvLayer
import Activators
import Activators_sec
from FC import FullyConnect
from FC_sec import FullyConnect_sec
from BN import BatchNorm
from BN_sec import BatchNorm_sec
from Logsoftmax import Logsoftmax
from Logsoftmax_sec import Logsoftmax_sec
from Deconv import Deconv
from Deconv_sec import Deconv_sec

'''
    验证：卷积计算Conv（成功）
    测试了多种条件（stride,padding,batch）变化下的卷积正确性
'''
def conv_test():
    bit_length = 32
    # (1,28,28)*(5,5,5)
    # x_numpy = np.random.randn(1,1,28,28).astype(np.float32)
    # w_numpy = np.random.randn(5,1,5,5).astype(np.float32)
    # b_numpy = np.random.randn(5).astype(np.float32)
    # # (1,28,28)*(5,5,5)
    # x_numpy_1 = np.random.randn(1,1,28,28).astype(np.float32)
    # x_numpy_2 = x_numpy-x_numpy_1
    # w_numpy_1 = np.random.randn(5,1,5,5).astype(np.float32)
    # w_numpy_2 = w_numpy-w_numpy_1
    # b_numpy_1 = np.random.randn(5).astype(np.float32)
    # b_numpy_2 = b_numpy-b_numpy_1

    ## (3,32,32)*(64,2,2)
    # x_numpy = np.random.randn(1,3,32,32).astype(np.float32)
    # w_numpy = np.random.randn(64,3,2,2).astype(np.float32)
    # b_numpy = np.random.randn(64).astype(np.float32)
    # x = torch.tensor(x_numpy, requires_grad=True)

    # x_numpy_1 = np.random.randn(1,3,32,32).astype(np.float32)
    # x_numpy_2 = x_numpy-x_numpy_1
    # w_numpy_1 = np.random.randn(64,3,2,2).astype(np.float32)
    # w_numpy_2 = w_numpy-w_numpy_1
    # b_numpy_1 = np.random.randn(64).astype(np.float32)
    # b_numpy_2 = b_numpy-b_numpy_1

    x_numpy = np.random.randn(1,32,32,32).astype(np.float32)
    w_numpy = np.random.randn(128,32,3,3).astype(np.float32)
    b_numpy = np.random.randn(128).astype(np.float32)
    x = torch.tensor(x_numpy, requires_grad=True)

    x_numpy_1 = np.random.randn(1,32,32,32).astype(np.float32)
    x_numpy_2 = x_numpy-x_numpy_1
    w_numpy_1 = np.random.randn(128,32,3,3).astype(np.float32)
    w_numpy_2 = w_numpy-w_numpy_1
    b_numpy_1 = np.random.randn(128).astype(np.float32)
    b_numpy_2 = b_numpy-b_numpy_1

    print('input_shape: ', x_numpy.shape)
    print('w_shape: ', w_numpy.shape)

    # padding=0, stride=2
    # cl1 = Conv_sec(1, 5, 5, 5, zero_padding=0, stride=2, method='SAME')
    # cl1 = Conv_sec(3, 64, 2, 2, zero_padding=0, stride=2, method='SAME')
    cl1 = Conv_sec(32, 128, 3, 3, zero_padding=0, stride=2, method='SAME')
    cl_ori = ConvLayer(1, 5, 5, 5, zero_padding=1, stride=2, method='SAME')
    cl_tensor = torch.nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=1)
    ## 设置参数
    cl_ori.set_weight(Parameter(w_numpy, requires_grad=True))
    cl_ori.set_bias(Parameter(b_numpy, requires_grad=True))
    cl1.set_weight_1(Parameter(w_numpy_1, requires_grad=True))
    cl1.set_bias_1(Parameter(b_numpy_1, requires_grad=True))
    cl1.set_weight_2(Parameter(w_numpy_2, requires_grad=True))
    cl1.set_bias_2(Parameter(b_numpy_2, requires_grad=True))

    # print('param_error: \n', w_numpy-(w_numpy_1+w_numpy_2))
    # print('param_error: \n', cl_ori.weights.data-(cl1.weights_1.data+cl1.weights_2.data))

    '''前向传播'''
    # start_time_tensor = time.time()
    # conv_out = cl_tensor(x)
    # end_time_tensor = time.time()
    # start_time = time.time()
    # conv_out = cl_ori.forward(x_numpy)
    # end_time = time.time()

    test_num = 10
    time_avg = 0
    for i in range(test_num):
        start_time_sec = time.time()
        conv_out_1, conv_out_2 = cl1.forward(x_numpy_1, x_numpy_2)
        end_time_sec = time.time()
        time_avg+=(end_time_sec-start_time_sec)*1000
    print('time avg sec: \n', time_avg/test_num)
    

    # print('conv_out: \n',conv_out[0][0])
    # print('conv_out_recover: \n',(conv_out_1+conv_out_2)[0][0])
    # print('forward_error: \n', conv_out-(conv_out_1+conv_out_2))
    # print('time consume sec: \n', (end_time_sec-start_time_sec)*1000)
    # print('time consume ori: \n', (end_time-start_time)*1000)
    # print('time consume tensor cpu: \n', (end_time_tensor-start_time_tensor)*1000)

'''Square激活函数测试'''
def square_test():
    bit_length = 32
    x_numpy = np.random.randn(1,1,28,28).astype(np.float64)
    x_numpy_1 = np.random.randn(1,1,28,28).astype(np.float64)
    x_numpy_2 = x_numpy-x_numpy_1
    sq = Activators.Square()
    sq_sec = Activators_sec.Square(bit_length)

    sq_out = sq.forward(x_numpy)
    sq_out_1, sq_out_2 = sq_sec.forward(x_numpy_1, x_numpy_2)

    print('x_numpy: ',x_numpy)
    print('sq_out: ',sq_out)
    print('error: ',sq_out-(sq_out_1+sq_out_2))

'''ReLU函数测试'''
def relu_test():
    bit_length = 32
    width = 169
    height = 169
    x_numpy = np.random.randn(1,1,width,height).astype(np.float64)
    x_numpy_1 = np.random.randn(1,1,width,height).astype(np.float64)
    x_numpy_2 = x_numpy-x_numpy_1
    
    # relu = Activators.ReLU()
    relu_sec = Activators_sec.ReLU(bit_length=bit_length)

    # relu_out = relu.forward(x_numpy)
    # relu_out1, relu_out2 = relu_sec.forward(x_numpy_1,x_numpy_2)

    test_num = 10
    time_avg = 0
    for i in range(test_num):
        start_time = time.time()
        relu_out1, relu_out2, dfc_time = relu_sec.forward(x_numpy_1,x_numpy_2)
        end_time = time.time()
        time_avg+=(end_time-start_time)*1000-dfc_time*60*1000
    print('time avg sec: \n', time_avg/test_num)

    # print('x_numpy: ',x_numpy)
    # print('sq_out: ',relu_out)
    # print('error: ',relu_out-(relu_out1+relu_out2))

def lrelu_test():
    bit_length = 64
    width = 5
    height = 5
    x_numpy = np.random.randn(1,1,width,height).astype(np.float64)
    x_numpy_1 = np.random.randn(1,1,width,height).astype(np.float64)
    x_numpy_2 = x_numpy-x_numpy_1
    print('x: ',x_numpy)
    print('x: ',x_numpy)

    lrelu = Activators.LeakyReLU(0.2)
    lrelu_sec = Activators_sec.LeakyReLU(0.2,bit_length=bit_length)

    lrelu_out = lrelu.forward(x_numpy)
    lrelu_out1, lrelu_out2, dfc_time = lrelu_sec.forward(x_numpy_1,x_numpy_2)

    print('lrelu: ',lrelu_out)
    print('lrelu_sec: ',lrelu_out1+lrelu_out2)

    print('error: ',lrelu_out-(lrelu_out1+lrelu_out2))


def fc_test():
    bit_length = 32
    batch_size = 1
    in_num = 580
    out_num = 580
    x_numpy = np.random.randn(batch_size, in_num).astype(np.float64)
    w_numpy = np.random.randn(in_num, out_num).astype(np.float64)
    b_numpy = np.random.randn(out_num).astype(np.float64)

    ## 准备秘密分享secret sharing
    x_numpy_1 = np.random.randn(batch_size, in_num).astype(np.float64)
    x_numpy_2 = x_numpy-x_numpy_1
    w_numpy_1 = np.random.randn(in_num, out_num).astype(np.float64)
    w_numpy_2 = w_numpy-w_numpy_1
    b_numpy_1 = np.random.randn(out_num).astype(np.float64)
    b_numpy_2 = b_numpy-b_numpy_1

    fc = FullyConnect(in_num, out_num)
    fc_sec = FullyConnect_sec(in_num, out_num, bit_length=bit_length)

    # 设置参数
    fc.set_weight(Parameter(w_numpy, requires_grad=True))
    fc_sec.set_weight_1(Parameter(w_numpy_1, requires_grad=True))
    fc_sec.set_weight_2(Parameter(w_numpy_2, requires_grad=True))
    fc.set_bias(Parameter(b_numpy, requires_grad=True))
    fc_sec.set_bias_1(Parameter(b_numpy_1, requires_grad=True))
    fc_sec.set_bias_2(Parameter(b_numpy_2, requires_grad=True))

    # fc_out = fc.forward(x_numpy)
    # fc_out_1, fc_out_2 = fc_sec.forward(x_numpy_1, x_numpy_2)
    # print('error: \n', fc_out-(fc_out_1+fc_out_2))

    test_num = 10
    time_avg = 0
    for i in range(test_num):
        start_time_sec = time.time()
        fc_out_1, fc_out_2 = fc_sec.forward(x_numpy_1, x_numpy_2)
        end_time_sec = time.time()
        time_avg+=(end_time_sec-start_time_sec)*1000
    print('time avg sec: \n', time_avg/test_num)
    

def bn_test():
    bit_length = 64
    width = 8
    height = 8
    channel = 256
    
    x_numpy = np.random.randn(1,channel,height,width).astype(np.float64)
    x_numpy_1 = np.random.randn(1,channel,height,width).astype(np.float64)
    x_numpy_2 = x_numpy - x_numpy_1
    # print('input: ',x_numpy)

    w_numpy = np.random.normal(1.0, 0.02, size=(channel)).astype(np.float64)
    b_numpy = np.zeros(channel).astype(np.float64)

    bn = BatchNorm(channel)
    bn_sec = BatchNorm_sec(channel)
    # 设置参数
    bn.set_gamma(Parameter(w_numpy, requires_grad=True))
    bn.set_beta(Parameter(b_numpy, requires_grad=True))
    bn_sec.set_gamma(Parameter(w_numpy, requires_grad=True))
    bn_sec.set_beta(Parameter(b_numpy, requires_grad=True))

    bn_out = bn.forward(x_numpy)

    bn_out_sec_1, bn_out_sec_2 = bn_sec.forward(x_numpy_1, x_numpy_2)
    
    # print('error sum: ',bn_out-(bn_out_sec_1+bn_out_sec_2))
    
    test_num = 10
    time_avg = 0
    for i in range(test_num):
        start_time_sec = time.time()
        bn_out_sec_1, bn_out_sec_2 = bn_sec.forward(x_numpy_1, x_numpy_2)
        end_time_sec = time.time()
        time_avg+=(end_time_sec-start_time_sec)*1000
    print('time avg sec: \n', time_avg/test_num)
    

def sigmoid_test():
    bit_length = 64
    width = 128
    height = 128
    
    # x_numpy = np.random.randn(1,1,height,width).astype(np.float64)
    # x_numpy_1 = np.random.randn(1,1,height,width).astype(np.float64)
    # x_numpy_2 = x_numpy - x_numpy_1

    ## 范围测试
    x_numpy = np.random.uniform(-1,1, size=(height,width)).astype(np.float64)
    x_numpy_1 = np.random.uniform(-10,-2, size=(height,width)).astype(np.float64)
    x_numpy_2 = x_numpy - x_numpy_1
    # print('input: ', x_numpy)

    sig_sec = Activators_sec.Sigmoid_CE_sec(bit_length=bit_length)
    sig = Activators.Sigmoid_CE()
    out = sig.forward(x_numpy)
    out_1, out_2 = sig_sec.forward(x_numpy_1, x_numpy_2)
    # print('result: ', out)
    # print('result_sec: ', out_1+out_2)
    # print('error: ', out-out_1-out_2)

    ## 速度测试
    
    test_num = 10
    time_avg = 0
    for i in range(test_num):
        start_time_sec = time.time()
        out_1, out_2 = sig_sec.forward(x_numpy_1, x_numpy_2)
        end_time_sec = time.time()
        time_avg+=(end_time_sec-start_time_sec)*1000
    print('time avg sec: \n', time_avg/test_num)
    

def logsoftmax_test():
    bit_length = 64
    batch = 5
    class_num = 3
    
    x_numpy = np.random.randn(batch,class_num).astype(np.float64)
    x_numpy_1 = np.random.randn(batch,class_num).astype(np.float64)
    x_numpy_2 = x_numpy - x_numpy_1

    # 初始化LogSoftmax类
    print('x: ', x_numpy)
    logsoftmax = Logsoftmax()
    output = logsoftmax.forward(x_numpy)

    logsoftmax_sec = Logsoftmax_sec(bit_length)
    output_1, output_2 = logsoftmax_sec.forward(x_numpy_1, x_numpy_2)
    print('error: ', output-(output_1+output_2))

def deconv_test():
    bit_length = 64
    ## share 没问题
    w_numpy = np.random.randn(3,5,3,3).astype(np.float64)
    w_numpy_1 = np.random.randn(3,5,3,3).astype(np.float64)
    w_numpy_2 = w_numpy - w_numpy_1
    b_numpy = np.random.randn(5).astype(np.float64)
    b_numpy_1 = np.random.randn(5).astype(np.float64)
    b_numpy_2 = b_numpy - b_numpy_1
    x_numpy = np.ones((2,3,2,2)).astype(np.float64)
    x_numpy_1 = np.ones((2,3,2,2)).astype(np.float64)
    x_numpy_2 = x_numpy - x_numpy_1

    # print('w: ', w_numpy)
    # print('w: ', w_numpy_1)

    decl_numpy = Deconv(3, out_channels=5, filter_size=3,  zero_padding=0, stride=1)
    decl_numpy.set_weight(Parameter(w_numpy, requires_grad=True))
    decl_numpy.set_bias(Parameter(b_numpy, requires_grad=True))
    deconv_out_numpy = decl_numpy.forward(x_numpy)

    dec_sec = Deconv_sec(3, out_channels=5, filter_size=3,  zero_padding=0, stride=1, bit_length=bit_length)
    dec_sec.set_weight_1(Parameter(w_numpy_1, requires_grad=True))
    dec_sec.set_bias_1(Parameter(b_numpy_1, requires_grad=True))
    dec_sec.set_weight_2(Parameter(w_numpy_2, requires_grad=True))
    dec_sec.set_bias_2(Parameter(b_numpy_2, requires_grad=True))
    dec_sec_out_1, dec_sec_out_2 = dec_sec.forward(x_numpy_1, x_numpy_2)

    # print('error: ', deconv_out_numpy-(dec_sec_out_1+dec_sec_out_2))
    '''
    test_num = 10
    time_avg = 0
    for i in range(test_num):
        start_time_sec = time.time()
        dec_sec_out_1, dec_sec_out_2 = dec_sec.forward(x_numpy_1, x_numpy_2)
        end_time_sec = time.time()
        time_avg+=(end_time_sec-start_time_sec)*1000
    print('time avg sec: \n', time_avg/test_num)
    '''

def tanh_test():
    bit_length = 64
    x_numpy = np.random.randn(6,1).astype(np.float64)
    x_numpy_1 = np.random.randn(6,1).astype(np.float64)*100
    x_numpy_2 = x_numpy-x_numpy_1
    print('input: ',x_numpy)
    print('input_1: ',x_numpy_1)
    print('input_2: ',x_numpy_2)
    
    tanh = Activators.Tanh()
    t_out = tanh.forward(x_numpy)

    tanh_sec = Activators_sec.Tanh_sec(bit_length=bit_length)
    t_out_1, t_out_2 = tanh_sec.forward(x_numpy_1, x_numpy_2)

    print('error tanh: ', t_out-(t_out_1+t_out_2))

if __name__ == '__main__':
    # conv_test()
    # square_test()
    # relu_test()
    # lrelu_test()
    # fc_test()
    # bn_test()
    sigmoid_test()
    # logsoftmax_test()
    # deconv_test()
    # tanh_test()
