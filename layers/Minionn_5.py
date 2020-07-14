'''
    测试MiniONN 5层结构的预测情况
'''

import random
import numpy as np
from Module import Module
from Optimizer_func import Adam
from Parameter import Parameter
from Conv_sec import Conv_sec
from FC_sec import FullyConnect_sec
import Activators_sec
from Logsoftmax import Logsoftmax
from Logsoftmax_sec import Logsoftmax_sec
from Loss import NLLLoss
from Loss_sec import NLLLoss_sec

import torch
from torchvision import datasets,transforms
import os
import time

n_epochs = 2 # 训练轮数
n_epochs_pre = 0 # 预训练轮数（加载已经训练好的模型时可以更新）
batch_size = 64
bit_length = 64

'''
    网络结构：
    1、conv: in_size=1x28x28, out_channel=5*196 k_size=5x5, stride=(2,2), padding=2
    2、ReLU Activation
    3、FC: in_size=980x1, out_size=100x1
    4、ReLU Activation
    5、FC: in_size=100x1, out_size=10x1
'''
class Minionn_fivelayer(Module):
    def __init__(self, in_dim, n_class):
        super(Minionn_fivelayer, self).__init__()

        self.conv = Conv_sec(in_dim, 5,5,5, zero_padding=2, stride=2, method='SAME',bit_length=bit_length)
        self.relu1 = Activators_sec.ReLU(bit_length=bit_length)
        self.fc1 = FullyConnect_sec(980, 100,bit_length=bit_length)
        self.relu2 = Activators_sec.ReLU(bit_length=bit_length)
        self.fc2 = FullyConnect_sec(100, n_class,bit_length=bit_length)
        self.logsoftmax = Logsoftmax_sec(bit_length=bit_length)

    def forward(self, x1, x2):
        in_size = x1.shape[0]
        # start_time_sum = time.time()

        # start_conv = time.time()
        out_c1, out_c2 = self.conv.forward(x1, x2)
        # end_conv = time.time()

        out_11, out_12, dfc_time1 = self.relu1.forward(out_c1, out_c2)

        # start_fc1 = time.time()
        self.conv_out_shape = out_11.shape
        # print('out1shape: ',self.conv_out_shape)
        out_11 = out_11.reshape(in_size, -1) # 将输出拉成一行
        out_12 = out_12.reshape(in_size, -1) # 将输出拉成一行
        out_fc11, out_fc12 = self.fc1.forward(out_11, out_12)
        # end_fc1 = time.time()
        out_21, out_22, dfc_time2= self.relu2.forward(out_fc11, out_fc12)
        # start_fc2 = time.time()
        out_31, out_32 = self.fc2.forward(out_21, out_22)
        # end_fc2 = time.time()

        # end_time_sum = time.time()

        # print('time consume: ', (end_conv-start_conv)*1000+(end_fc1-start_fc1)*1000+(end_fc1-start_fc1)*1000+self.relu1.offline_time+self.relu1.online_time+self.relu2.offline_time+self.relu2.online_time)
        # print('time consume sum: ', (end_time_sum-start_time_sum)*1000)
        out_logsoftmax_1, out_logsoftmax_2 = self.logsoftmax.forward(out_31, out_32)
        return out_logsoftmax_1, out_logsoftmax_2, dfc_time1+dfc_time2
        # out_logsoftmax = self.logsoftmax.forward(out_31+out_32)
        # return out_logsoftmax

    def backward(self, dy1, dy2):
        dy_logsoftmax_1, dy_logsoftmax_2 = self.logsoftmax.gradient(dy1, dy2)
        dy_f31, dy_f32 = self.fc2.gradient(dy_logsoftmax_1, dy_logsoftmax_2)
        dy_relu21, dy_relu22 = self.relu2.gradient(dy_f31, dy_f32)
        dy_f21, dy_f22 = self.fc1.gradient(dy_relu21, dy_relu22)

        dy_f21 = dy_f21.reshape(self.conv_out_shape)
        dy_f22 = dy_f22.reshape(self.conv_out_shape)

        dy_relu11, dy_relu12 = self.relu1.gradient(dy_f21, dy_f22)
        self.conv.gradient(dy_relu11, dy_relu12)

'''只做预测操作'''
if __name__ == '__main__': 
    """处理MNIST数据集"""
    train_dataset = datasets.MNIST('../data/',download=True,train=True,transform=transforms.Compose([
                                   transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                               ]))
    test_dataset = datasets.MNIST('../data/',download=True,train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                              ]))
    print('traindata_len: \n',len(train_dataset))
    print('traindata_len: \n',len(test_dataset))
    # 构建数据集迭代器
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    """初始化CNN网络"""
    Minionn_5 = Minionn_fivelayer(in_dim=1, n_class=10)
    print('Minionn_5: \n', Minionn_5)

    loss_fn = NLLLoss_sec()
    lr = 1e-3# Adam优化器的学习率(正常是1e-3)
    beta1 = 0.9 # Adam优化器的参数（需要调整试试？）(正常是0.5)
    optimizer = Adam(Minionn_5.parameters(), learning_rate=lr, betas=(beta1, 0.999)) # 测试一下Adam优化器
    # 加载预训练的模型并
    
    pre_module_path = "../model_save/Minionn_5/Minionn_5_sec_parameters-4.pkl"
    params = torch.load(pre_module_path)
    # params_ori = params['state_dict']
    Minionn_5.load_state_dict(params['state_dict']) # 加载模型
    n_epochs_pre = params['epoch']
    print('pre epoch: ', n_epochs_pre)
    print('pre train_acc: ', params['train_acc'])
    
    '''
        模型参数：
        conv.weights, conv.bias
        fc1.weights, fc1.bias
        fc2.weights, fc2.bias
    '''
    '''
    # 将模型拆分，重新赋值
    ## Conv
    conv_weights = params_ori['conv.weights']
    conv_bias = params_ori['conv.bias']
    conv_weights_shape = conv_weights.shape
    conv_bias_shape = conv_bias.shape
    conv_weights_1 = np.random.randn(conv_weights_shape[0], conv_weights_shape[1], conv_weights_shape[2], conv_weights_shape[3])
    conv_bias_1 = np.random.randn(conv_bias_shape[0])
    conv_weights_2 = conv_weights - conv_weights_1
    conv_bias_2 = conv_bias - conv_bias_1
    Minionn_5.conv.set_weight_1(Parameter(conv_weights_1, requires_grad=True))
    Minionn_5.conv.set_weight_2(Parameter(conv_weights_2, requires_grad=True))
    Minionn_5.conv.set_bias_1(Parameter(conv_bias_1, requires_grad=True))
    Minionn_5.conv.set_bias_2(Parameter(conv_bias_2, requires_grad=True))

    # print('conv.shape: ', conv_weights.shape)
    # print('conv1.shape: ', conv_weights_1.shape)
    # print('conv2.shape: ', conv_weights_2.shape)
    # print('conv.bias: \n', conv_bias)
    # print('conv_bias_1+conv_bias_2: \n', Minionn_5.conv.bias_1.data+Minionn_5.conv.bias_2.data)

    ## fc1
    fc1_weights = params_ori['fc1.weights']
    fc1_bias = params_ori['fc1.bias']
    fc1_weights_shape = fc1_weights.shape
    fc1_bias_shape = fc1_bias.shape
    fc1_weights_1 = np.random.randn(fc1_weights_shape[0], fc1_weights_shape[1])
    fc1_bias_1 = np.random.randn(fc1_bias_shape[0])
    fc1_weights_2 = fc1_weights - fc1_weights_1
    fc1_bias_2 = fc1_bias - fc1_bias_1
    Minionn_5.fc1.set_weight_1(Parameter(fc1_weights_1, requires_grad=True))
    Minionn_5.fc1.set_weight_2(Parameter(fc1_weights_2, requires_grad=True))
    Minionn_5.fc1.set_bias_1(Parameter(fc1_bias_1, requires_grad=True))
    Minionn_5.fc1.set_bias_2(Parameter(fc1_bias_2, requires_grad=True))

    # print('fc1.bias: \n', fc1_bias)
    # print('fc1_bias_1+fc1_bias_2: \n', Minionn_5.fc1.bias_1.data+Minionn_5.fc1.bias_2.data)

    ## fc2
    fc2_weights = params_ori['fc2.weights']
    fc2_bias = params_ori['fc2.bias']
    fc2_weights_shape = fc2_weights.shape
    fc2_bias_shape = fc2_bias.shape
    fc2_weights_1 = np.random.randn(fc2_weights_shape[0], fc2_weights_shape[1])
    fc2_bias_1 = np.random.randn(fc2_bias_shape[0])
    fc2_weights_2 = fc2_weights - fc2_weights_1
    fc2_bias_2 = fc2_bias - fc2_bias_1
    Minionn_5.fc2.set_weight_1(Parameter(fc2_weights_1, requires_grad=True))
    Minionn_5.fc2.set_weight_2(Parameter(fc2_weights_2, requires_grad=True))
    Minionn_5.fc2.set_bias_1(Parameter(fc2_bias_1, requires_grad=True))
    Minionn_5.fc2.set_bias_2(Parameter(fc2_bias_2, requires_grad=True))
    # print('fc2.bias: \n', fc2_bias)
    # print('fc2_bias_1+fc2_bias_2: \n', Minionn_5.fc2.bias_1.data+Minionn_5.fc2.bias_2.data)

    # print(params['state_dict']['conv.weights'])
    # print(type(params['state_dict']['conv.weights']))
    '''

    for epoch in range(n_epochs):
        # break
        running_loss = 0.0
        running_correct = 0
        dfc_time_sum = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-"*10)
        
        start_time = time.time()
        for t, (data, target) in enumerate(train_loader):
            # 将tensor类型的data转为numpy
            data = data.detach().numpy()
            target = target.detach().numpy()
            ## 拆分数据成secret shares
            data_shape = data.shape
            data_1 = np.random.randn(data_shape[0], data_shape[1], data_shape[2], data_shape[3])
            data_2 = data-data_1
            ## 拆分标签成secret shares
            target_shape = target.shape # [batchsize,]
            target_1 = np.random.randint(1,10,size=target_shape)
            target_2 = target-target_1
            ## 安全预测
            # pred = Minionn_5.forward(data_1,data_2) # pred=[x1,x2,...,xn]
            pred_1, pred_2, dfc_time = Minionn_5.forward(data_1,data_2) # pred=[x1,x2,...,xn]
            dfc_time_sum += dfc_time
            # output = np.argmax(pred, axis=1) 
            output = np.argmax(pred_1+pred_2, axis=1) # 统计输出结果就使用明文吧= =
            # loss = loss_fn.cal_loss(pred, target)
            loss_1, loss_2 = loss_fn.cal_loss(pred_1, pred_2, target_1, target_2)
            optimizer.zero_grad()
            # dy_loss = loss_fn.gradient()
            dy_loss_1, dy_loss_2 = loss_fn.gradient()
            # print('dy_loss: \n', dy_loss)

            ## 拆分计算的损失，进行反向传播，训练网络
            # Minionn_5.backward(dy_loss)
            Minionn_5.backward(dy_loss_1, dy_loss_2)
            optimizer.step()

            # 计算总损失
            running_loss += (loss_1+loss_2)
            running_correct += sum(output == target) 
            
            if t%5==0 and t!=0:
                end_time = time.time()
                print("Step/Epoch:{}/{}, Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Calculate time: {:.4f}min".format(t, epoch, running_loss/(t*batch_size), 100.0*running_correct/(t*batch_size), (end_time-start_time)/60-dfc_time_sum))
                # print('dfc_time: ',dfc_time_sum)

        # 存储安全模型
        checkpoint_path = "../model_save/Minionn_5/Minionn_5_sec_parameters-"+str(n_epochs+n_epochs_pre)+".pkl"
        torch.save({'epoch':n_epochs+n_epochs_pre, 'state_dict':Minionn_5.state_dict(), 'train_acc':100*running_correct/len(train_dataset) }, checkpoint_path)
    

    '''使用测试集做预测'''
    '''
    testing_correct = 0
    for t, (data, target) in enumerate(test_loader):
        x_test = data.detach().numpy()
        ## 对测试集数据做预处理，拆分为secret sharing
        x_test_shape = x_test.shape
        x_test_1 = np.random.randn(x_test_shape[0], x_test_shape[1], x_test_shape[2], x_test_shape[3])
        x_test_2 = x_test-x_test_1
        # print('x_test: \n',x_test[0])
        # print('x_test_1+x_test_2: \n',(x_test_1+x_test_2)[0]-x_test[0])
        # print('input.shape: ', x_test_1.shape)
        
        y_test = target.detach().numpy()
        pred = Minionn_5.forward(x_test_1, x_test_2)
        # print('pred: ',pred)
        output = np.argmax(pred, axis=1)
        testing_correct += sum(output == y_test) 
        print('round: ',t)
        break
    # print("Test Accuracy is:{:.4f}%".format(100.0*testing_correct/len(test_dataset)))
    print("Test Accuracy is:{:.4f}%".format(100.0*testing_correct/batch_size))
    '''