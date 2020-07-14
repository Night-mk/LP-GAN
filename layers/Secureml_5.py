'''
    测试CryptoNet 5层结构的安全训练和预测
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

n_epochs = 1 # 训练轮数
n_epochs_pre = 0 # 预训练轮数（加载已经训练好的模型时可以更新）
batch_size = 128
bit_length = 64

'''
    网络结构：
    1、FC: in_size=784x1, out_size=128x1
    2、Square Activation
    3、Pool: in_size=845x1, out_size=100x1，其实就是个FC？
    4、Square Activation
    5、FC: in_size=100x1, out_size=10x1
'''
class Secureml_fivelayer(Module):
    def __init__(self, in_dim, n_class):
        super(Secureml_fivelayer, self).__init__()

        self.fc0 = FullyConnect_sec(784,128,bit_length=bit_length)
        self.sq1 = Activators_sec.Square(bit_length=bit_length)
        self.fc1 = FullyConnect_sec(128, 128, bit_length=bit_length)
        self.sq2 = Activators_sec.Square(bit_length=bit_length)
        self.fc2 = FullyConnect_sec(128, n_class, bit_length=bit_length)
        self.logsoftmax = Logsoftmax_sec(bit_length=bit_length)
        # self.logsoftmax = Logsoftmax()

    def forward(self, x1, x2):
        in_size = x1.shape[0]
        x1 = x1.reshape(in_size, -1)
        x2 = x2.reshape(in_size, -1)

        # start_time = time.time()
        out_c1, out_c2 = self.fc0.forward(x1, x2)
        out_11, out_12 = self.sq1.forward(out_c1, out_c2)
        out_fc11, out_fc12 = self.fc1.forward(out_11, out_12)
        out_21, out_22= self.sq2.forward(out_fc11, out_fc12)
        out_31, out_32 = self.fc2.forward(out_21, out_22)
        # end_time = time.time()
        # print('time consume: ', (end_time-start_time)*1000)
        
        # out_logsoftmax = self.logsoftmax.forward(out_31+out_32)
        # print('out_3: ',out_31+out_32)
        out_logsoftmax_1, out_logsoftmax_2 = self.logsoftmax.forward(out_31, out_32)
        return out_logsoftmax_1, out_logsoftmax_2
        # return out_logsoftmax

    def backward(self, dy1, dy2):
        # dy_logsoftmax = self.logsoftmax.gradient(dy)
        dy_logsoftmax_1, dy_logsoftmax_2 = self.logsoftmax.gradient(dy1, dy2)
        # 生成shares
        # dy_logsoftmax_1 = np.random.uniform(0,1, dy_logsoftmax.shape)
        # dy_logsoftmax_2 = dy_logsoftmax - dy_logsoftmax_1
        dy_f31, dy_f32 = self.fc2.gradient(dy_logsoftmax_1, dy_logsoftmax_2)
        
        dy_sq21, dy_sq22 = self.sq2.gradient(dy_f31, dy_f32)
        dy_f21, dy_f22 = self.fc1.gradient(dy_sq21, dy_sq22)
        
        dy_sq11, dy_sq12 = self.sq1.gradient(dy_f21, dy_f22)
        self.fc0.gradient(dy_sq11, dy_sq12)
        # dy_f3 = self.fc2.gradient(dy_logsoftmax)
        # dy_f2 = self.fc1.gradient(self.sq2.gradient(dy_f3))
        # self.fc0.gradient(self.sq1.gradient(dy_f2))

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
    print('testdata_len: \n',len(test_dataset))
    # 构建数据集迭代器
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    """初始化CNN网络"""
    Secureml_5 = Secureml_fivelayer(in_dim=1, n_class=10)
    print('Secureml_5: \n', Secureml_5)
    # print('param_sec: ',Secureml_5.parameters())

    """构建优化器"""
    # loss_fn = NLLLoss()
    loss_fn = NLLLoss_sec(bit_length=bit_length)
    lr = 1e-3# Adam优化器的学习率
    beta1 = 0.5 # Adam优化器的参数（需要调整试试？）
    optimizer = Adam(Secureml_5.parameters(), learning_rate=lr, betas=(beta1, 0.999)) # 测试一下Adam优化器
    # 加载预训练的模型并预测
    '''
    pre_module_path = "../model_save/Secureml_5/Secureml_5_parameters-1.pkl"
    params = torch.load(pre_module_path)
    params_ori = params['state_dict']
    '''
    '''
        模型参数：
        fc0.weights, fc0.bias
        fc1.weights, fc1.bias
        fc2.weights, fc2.bias
    '''
    # 将模型拆分，重新赋值
    '''
    ## fc0
    fc0_weights = params_ori['fc0.weights']
    fc0_bias = params_ori['fc0.bias']
    fc0_weights_shape = fc0_weights.shape
    fc0_bias_shape = fc0_bias.shape
    fc0_weights_1 = np.random.randn(fc0_weights_shape[0], fc0_weights_shape[1])
    fc0_bias_1 = np.random.randn(fc0_bias_shape[0])
    fc0_weights_2 = fc0_weights - fc0_weights_1
    fc0_bias_2 = fc0_bias - fc0_bias_1
    Secureml_5.fc0.set_weight_1(Parameter(fc0_weights_1, requires_grad=True))
    Secureml_5.fc0.set_weight_2(Parameter(fc0_weights_2, requires_grad=True))
    Secureml_5.fc0.set_bias_1(Parameter(fc0_bias_1, requires_grad=True))
    Secureml_5.fc0.set_bias_2(Parameter(fc0_bias_2, requires_grad=True))
    # print('fc0.bias: \n', fc0_bias)
    # print('fc0_bias_1+fc0_bias_2: \n', Secureml_5.fc0.bias_1.data+Secureml_5.fc0.bias_2.data)
    
    ## fc1
    fc1_weights = params_ori['fc1.weights']
    fc1_bias = params_ori['fc1.bias']
    fc1_weights_shape = fc1_weights.shape
    fc1_bias_shape = fc1_bias.shape
    fc1_weights_1 = np.random.randn(fc1_weights_shape[0], fc1_weights_shape[1])
    fc1_bias_1 = np.random.randn(fc1_bias_shape[0])
    fc1_weights_2 = fc1_weights - fc1_weights_1
    fc1_bias_2 = fc1_bias - fc1_bias_1
    Secureml_5.fc1.set_weight_1(Parameter(fc1_weights_1, requires_grad=True))
    Secureml_5.fc1.set_weight_2(Parameter(fc1_weights_2, requires_grad=True))
    Secureml_5.fc1.set_bias_1(Parameter(fc1_bias_1, requires_grad=True))
    Secureml_5.fc1.set_bias_2(Parameter(fc1_bias_2, requires_grad=True))
    # print('fc1.bias: \n', fc1_bias)
    # print('fc1_bias_1+fc1_bias_2: \n', Secureml_5.fc1.bias_1.data+Secureml_5.fc1.bias_2.data)

    ## fc2
    fc2_weights = params_ori['fc2.weights']
    fc2_bias = params_ori['fc2.bias']
    fc2_weights_shape = fc2_weights.shape
    fc2_bias_shape = fc2_bias.shape
    fc2_weights_1 = np.random.randn(fc2_weights_shape[0], fc2_weights_shape[1])
    fc2_bias_1 = np.random.randn(fc2_bias_shape[0])
    fc2_weights_2 = fc2_weights - fc2_weights_1
    fc2_bias_2 = fc2_bias - fc2_bias_1
    Secureml_5.fc2.set_weight_1(Parameter(fc2_weights_1, requires_grad=True))
    Secureml_5.fc2.set_weight_2(Parameter(fc2_weights_2, requires_grad=True))
    Secureml_5.fc2.set_bias_1(Parameter(fc2_bias_1, requires_grad=True))
    Secureml_5.fc2.set_bias_2(Parameter(fc2_bias_2, requires_grad=True))
    # print('fc2.bias: \n', fc2_bias)
    # print('fc2_bias_1+fc2_bias_2: \n', Secureml_5.fc2.bias_1.data+Secureml_5.fc2.bias_2.data)
    '''
    # Secureml_5.load_state_dict(params['state_dict']) # 加载模型
    # n_epochs_pre = params['epoch']

    '''使用训练集训练'''
    """迭代训练"""
    
    for epoch in range(n_epochs):
        # break
        running_loss = 0.0
        running_correct = 0
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
            # pred = Secureml_5.forward(data_1,data_2) # pred=[x1,x2,...,xn]
            pred_1, pred_2 = Secureml_5.forward(data_1,data_2) # pred=[x1,x2,...,xn]
            # output = np.argmax(pred, axis=1) 
            output = np.argmax(pred_1+pred_2, axis=1) # 统计输出结果就使用明文吧= =
            # loss = loss_fn.cal_loss(pred, target)
            loss_1, loss_2 = loss_fn.cal_loss(pred_1, pred_2, target_1, target_2)
            optimizer.zero_grad()
            # dy_loss = loss_fn.gradient()
            dy_loss_1, dy_loss_2 = loss_fn.gradient()
            # print('dy_loss: \n', dy_loss)

            ## 拆分计算的损失，进行反向传播，训练网络
            # Secureml_5.backward(dy_loss)
            Secureml_5.backward(dy_loss_1, dy_loss_2)
            optimizer.step()

            # 计算总损失
            running_loss += (loss_1+loss_2)
            running_correct += sum(output == target) 
            
            if t%5==0 and t!=0:
                end_time = time.time()
                print("Step/Epoch:{}/{}, Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Calculate time: {:.4f}min".format(t, epoch,running_loss/(t*batch_size), 100.0*running_correct/(t*batch_size), (end_time-start_time)/60))
        
    '''使用测试集做预测'''
    '''
    testing_correct = 0
    for t, (data, target) in enumerate(test_loader):
        x_test = data.detach().numpy()
        ## 对测试集数据做预处理，拆分为secret sharing
        x_test_shape = x_test.shape
        x_test_1 = np.random.randn(x_test_shape[0], x_test_shape[1], x_test_shape[2], x_test_shape[3])
        x_test_2 = x_test-x_test_1
        
        y_test = target.detach().numpy()
        pred_1, pred_2 = Secureml_5.forward(x_test_1, x_test_2)
        # pred = Secureml_5.forward(x_test_1, x_test_2)
        # print('pred: ',pred)
        output = np.argmax(pred_1+pred_2, axis=1)
        # output = np.argmax(pred, axis=1)
        testing_correct += sum(output == y_test) 
        break
        print('round: ',t)
    # print("Test Accuracy is:{:.4f}%".format(100.0*testing_correct/len(test_dataset)))
    print("Test Accuracy is:{:.4f}%".format(100.0*testing_correct/batch_size))
    '''    