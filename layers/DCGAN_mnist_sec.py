'''
    DCGAN_numpy.py 使用numpy实现DCGAN网络
'''

import random
import numpy as np
from Module import Module
from Optimizer_func import Adam
from Parameter import Parameter
from Conv import ConvLayer
from Conv_sec import Conv_sec
from Deconv import Deconv
from Deconv_sec import Deconv_sec
from Loss import BECLoss
import Activators
import Activators_sec
from BN import BatchNorm
from BN_sec import BatchNorm_sec

import torch
from torchvision import datasets,transforms
import os
import time

import matplotlib.pyplot as plt
import torchvision.utils as vutils 
# import Data_loader

# batch_size = 128 # 训练batch
batch_size = 64 # 训练batch
image_size = 32 # 训练图像size，默认64x64, MINST 28x28
nc = 3 # 输入图像通道数
nz = 100 # 生成器输入的噪声z维度, lantent vector size

ngf = 64 # 生成器特征图的数量
ndf = 64 # 判别器特征图的数量
num_epochs = 1 # 训练轮数
lr = 0.0002 # Adam优化器的学习率
beta1 = 0.5 # Adam优化器的参数
is_mnist = True
num_epochs_pre = 0 # 预训练轮数
bit_length = 64

"""判别器D网络"""
class Discriminator_sec(Module):
    def __init__(self):
        super(Discriminator_sec, self).__init__()
        # 输入1*28*28 MNIST
        # 1*28*28 -> 64*16*16
        self.conv1 = Conv_sec(nc, ndf, 4,4, zero_padding=1, stride=2,method='SAME', bias_required=False, bit_length=bit_length)
        self.lrelu1 = Activators_sec.LeakyReLU(0.2, bit_length=bit_length)

        # 64*16*16 -> 128*8*8
        self.conv2 = Conv_sec(ndf, ndf*2, 4,4, zero_padding=1, stride=2, method='SAME', bias_required=False, bit_length=bit_length)
        self.bn1 = BatchNorm_sec(ndf*2, bit_length=bit_length)
        self.lrelu2 = Activators_sec.LeakyReLU(0.2, bit_length=bit_length)

        # 128*8*8 -> 256*4*4
        self.conv3 = Conv_sec(ndf*2, ndf*4, 4,4, zero_padding=1, stride=2, method='SAME', bias_required=False, bit_length=bit_length)
        self.bn2 = BatchNorm_sec(ndf*4, bit_length=bit_length)
        self.lrelu3 = Activators_sec.LeakyReLU(0.2, bit_length=bit_length)

        # 256*4*4 -> 1*1
        self.conv4 = Conv_sec(ndf*4, 1, 4,4, zero_padding=0, stride=1, method='VALID', bias_required=False, bit_length=bit_length)
        self.sigmoid = Activators_sec.Sigmoid_CE_sec(bit_length=bit_length)

    def forward(self, x_input_1, x_input_2):
        # l1 = self.lrelu1.forward(self.conv1.forward(x_input))
        # l2 = self.lrelu2.forward(self.bn1.forward(self.conv2.forward(l1)))
        # l3 = self.lrelu3.forward(self.bn2.forward(self.conv3.forward(l2)))
        # l4 = self.conv4.forward(l3)
        # output_sigmoid = self.sigmoid.forward(l4)
        # return output_sigmoid
        start_l1 = time.time()
        conv11, conv12 = self.conv1.forward(x_input_1, x_input_2)
        # end_conv1 = time.time()
        lrelu11, lrelu12, dfc_time_1 = self.lrelu1.forward(conv11, conv12) # 还是relu最消耗时间= =
        # end_l1 = time.time()
        # print('conv1 (ms): ', (end_conv1-start_l1)*1000)
        # print('L_1 (ms): ', (end_l1-start_l1)*1000)

        conv21, conv22 = self.conv2.forward(lrelu11, lrelu12)
        bn11, bn12 = self.bn1.forward(conv21, conv22)
        lrelu21, lrelu22, dfc_time_2 = self.lrelu2.forward(bn11, bn12)
        
        conv31, conv32 = self.conv3.forward(lrelu21, lrelu22)
        bn21, bn22 = self.bn2.forward(conv31, conv32)
        lrelu31, lrelu32, dfc_time_3 = self.lrelu3.forward(bn21, bn22)

        conv41, conv42 = self.conv4.forward(lrelu31, lrelu32)
        # print('conv4: ', (conv41+conv42)[0][0])
        output_sig_1, output_sig_2 = self.sigmoid.forward(conv41, conv42)
        end_sig = time.time()

        print('sigmoid input shape: ', conv41.shape)
        print('dfc_time (s): ', (dfc_time_1+dfc_time_2+dfc_time_3)*60)
        print('total SD time (s): ', (end_sig-start_l1)-(dfc_time_1+dfc_time_2+dfc_time_3)*60)
        print('total SD time real (s): ', (end_sig-start_l1))

        return output_sig_1, output_sig_2
    
    def backward(self, dy):
        # print('dy.shape: ', dy.shape)
        dy_sigmoid = self.sigmoid.gradient(dy)
        # print('dy_sigmoid.shape: ', dy_sigmoid.shape)
        dy_l4 = self.conv4.gradient(dy_sigmoid)
        dy_l3 = self.conv3.gradient(self.bn2.gradient(self.lrelu3.gradient(dy_l4)))
        dy_l2 = self.conv2.gradient(self.bn1.gradient(self.lrelu2.gradient(dy_l3)))
        dy_l1 = self.conv1.gradient(self.lrelu1.gradient(dy_l2))
        # print('D_backward output shape: ',dy_l1.shape)
        return dy_l1

class Generator_sec(Module):
    def __init__(self):
        super(Generator_sec, self).__init__()
        # 构建反向传播网络组建
        # 输入Z=[100,]
        # 100*1 -> 256*4*4
        self.deconv1 = Deconv_sec(nz, ngf*4, 4, zero_padding=0, stride=1, method='VALID', bias_required=False, bit_length=bit_length)
        self.bn1 = BatchNorm_sec(ngf*4, bit_length=bit_length)
        self.relu1 = Activators_sec.ReLU(bit_length=bit_length)
        # 256*4*4 -> 128*8*8
        self.deconv2 = Deconv_sec(ngf*4, ngf*2, 4, zero_padding=1, stride=2, method='SAME', bias_required=False, bit_length=bit_length)
        self.bn2 = BatchNorm_sec(ngf*2, bit_length=bit_length)
        self.relu2 = Activators_sec.ReLU(bit_length=bit_length)
        # 128*8*8 -> 64*16*16
        self.deconv3 = Deconv_sec(ngf*2, ngf, 4, zero_padding=1, stride=2, method='SAME', bias_required=False, bit_length=bit_length)
        self.bn3 = BatchNorm_sec(ngf, bit_length=bit_length)
        self.relu3 = Activators_sec.ReLU(bit_length=bit_length)
        # 64*16*16 -> 1*32*32
        self.deconv4 = Deconv_sec(ngf, nc, 4, zero_padding=1, stride=2, method='SAME', bias_required=False, bit_length=bit_length)
        self.tanh = Activators_sec.Tanh_sec(bit_length=bit_length)

    def forward(self, x_input_1, x_input_2):
        # print('G input shape: ',x_input.shape)
        # l1 = self.relu1.forward(self.bn1.forward(self.deconv1.forward(x_input)))
        # l2 = self.relu2.forward(self.bn2.forward(self.deconv2.forward(l1)))
        # l3 = self.relu3.forward(self.bn3.forward(self.deconv3.forward(l2)))
        # l4 = self.deconv4.forward(l3)
        # output_tanh = self.tanh.forward(l4)

        start_l1 = time.time()
        deconv11, deconv12 = self.deconv1.forward(x_input_1, x_input_2)
        bn11, bn12 = self.bn1.forward(deconv11, deconv12)
        relu11, relu12, dfc_time_1 = self.relu1.forward(bn11, bn12) # 还是relu最消耗时间= =
        # end_l1 = time.time()
        # print('deconv comsume: ', (end_deconv-start_l1)*100)
        # print('deconv+bn comsume: ', (end_bn-start_l1)*1000)
        # print('relu online: ', (end_l1-start_relu)*1000-dfc_time_1*60*1000)
        # print('l1 consume: ', (end_l1-start_l1)*1000-dfc_time_1*60*1000)

        deconv21, deconv22 = self.deconv2.forward(relu11, relu12)
        bn21, bn22 = self.bn2.forward(deconv21, deconv22)
        relu21, relu22, dfc_time_2 = self.relu2.forward(bn21, bn22)
        
        deconv31, deconv32 = self.deconv3.forward(relu21, relu22)
        bn31, bn32 = self.bn3.forward(deconv31, deconv32)
        relu31, relu32, dfc_time_3 = self.relu3.forward(bn31, bn32)
        end_l3 = time.time()

        deconv41, deconv42 = self.deconv4.forward(relu31, relu32)
        start_tanh = time.time()
        output_tanh_1, output_tanh_2 = self.tanh.forward(deconv41, deconv42)
        end_tanh = time.time()
        # print('tanh_input_shape: ',deconv41.shape)
        # print('l1~l3 consume: ', (end_l3-start_l1)*1000-(dfc_time_1+dfc_time_2+dfc_time_3)*60*1000)
        # print('tanh consume (ms): ', (end_tanh-start_tanh)*1000)
        print('total_time (s): ',(end_tanh-start_l1)-(dfc_time_1+dfc_time_2+dfc_time_3)*60)
        print('total_time real (s): ',(end_tanh-start_l1))
        
        return output_tanh_1, output_tanh_2

    def backward(self, dy):
        dy_tanh = self.tanh.gradient(dy)
        dy_l4 = self.deconv4.gradient(dy_tanh)
        dy_l3 = self.deconv3.gradient(self.bn3.gradient(self.relu3.gradient(dy_l4)))
        dy_l2 = self.deconv2.gradient(self.bn2.gradient(self.relu2.gradient(dy_l3)))
        self.deconv1.gradient(self.bn1.gradient(self.relu1.gradient(dy_l2)))


'''批量初始化网络的权重'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # 卷积层和反卷积层设置没有bias参数
        m.weights.data = np.random.normal(0.0,0.02,size=m.weights.data.shape)
    elif classname.find('BatchNorm') != -1:
        # BN层初始化两组参数，weight=gamma，bias=beta
        m.gamma.data = np.random.normal(1.0, 0.02, size=m.gamma.data.shape)
        m.beta.data = np.zeros(m.beta.data.shape)
        
def test_dcgan():
    start_time = time.time()
    # 设置transform
    # cifar_transform = transforms.Compose([
    #     Data_loader.cifar_Resize((3,32,32))
    # ])
    """加载CIFAR-10数据集"""
    # root_dir = './data/cifar-10/cifar-10-python/cifar-10-batches-py'
    # cifar_train_dataset = Data_loader.CifarDataset(root_dir, transform=cifar_transform, train=True)
    # print('traindata_len: \n',len(cifar_train_dataset))
    # # 构建数据集迭代器
    # cifar_train_loader = torch.utils.data.DataLoader(cifar_train_dataset, batch_size=batch_size, shuffle=True)
    
    root_dir = '../data/cifar-10/cifar-10-image'
    cifar_train_dataset = datasets.ImageFolder(root=root_dir,
                               transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    print('cifar-traindata_len: \n',len(cifar_train_dataset))
    # 构建数据集迭代器
    cifar_train_loader = torch.utils.data.DataLoader(cifar_train_dataset, batch_size=batch_size, shuffle=True)

    mnist_train_dataset = datasets.MNIST('../data/',download=True,train=True,transform=transforms.Compose([
                                   transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                               ]))
    print('mnist_traindata_len: \n',len(mnist_train_dataset))
    # 构建数据集迭代器
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train_dataset,batch_size=batch_size,shuffle=True)
    """初始化网络、参数"""
    netG = Generator_sec()
    # netG.apply(weights_init)
    # print('netG: \n', netG)
    
    netD = Discriminator_sec()
    # netD.apply(weights_init)
    # print('netD: \n', netD)


    """构建优化器"""
    # 二进制交叉熵损失函数
    loss = BECLoss()
    # 噪声从标准正态分布（均值为0，方差为 1，即高斯白噪声）中随机抽取一组数
    fixed_noise = np.random.normal(0.0, 1.2, size=(batch_size, nz, 1,1)) # 用于G生成图像时固定的噪声初始化
    fixed_noise_1 = np.random.normal(0.0, 1.2, size=(batch_size, nz, 1,1)) # 用于G生成图像时固定的噪声初始化
    fixed_noise_2 = fixed_noise - fixed_noise_1
    # 定义真假样本标签
    real_label = 1
    fake_label = 0
    # 定义Adam优化器
    # optimizerD = Adam(netD.parameters(), learning_rate=lr, betas=(beta1, 0.999))
    # optimizerG = Adam(netG.parameters(), learning_rate=lr, betas=(beta1, 0.999))

    """训练模型，生成数据"""
    # 存储生成的图像
    img_list = []
    # 记录G和D的损失
    G_losses = []
    D_losses = []
    iters = 0

    """加载预训练的模型"""
    '''
    pre_module_path = "./model_save/DCGAN_numpy_parameters-cifar-Adam-2.pkl"
    params = torch.load(pre_module_path)
    netD.load_state_dict(params['D_state_dict']) # 加载模型
    netG.load_state_dict(params['G_state_dict']) # 加载模型
    num_epochs_pre = params['epoch']
    '''
    print('batch_size: ', batch_size)
    print('channel :', nc)

    '''测试SD判别时间'''
    '''
    ## 测试Cifar-10, CelebA数据集
    for t, (data, target) in enumerate(cifar_train_loader, 0):
        real_data = data.detach().numpy()
        print('real_data shape: \n',real_data[0].shape)
        data_shape = real_data.shape
        real_data_1 = np.random.uniform(0,1,size=data_shape)
        real_data_2 = real_data-real_data_1
        # 计算D前向传播值
        output_d_real_1, output_d_real_2 = netD.forward(real_data_1, real_data_2)
        print('inference: ', (output_d_real_1+output_d_real_2).reshape(-1))
        break
    return 0
    '''
    '''
    ## 测试MNIST数据集
    for t, (data, target) in enumerate(mnist_train_loader, 0):
        real_data = data.detach().numpy()
        print('real_data shape: \n',real_data[0].shape)
        # break
        ## MNIST 数据需要先从1x28x28填充到1x32x32
        if is_mnist:
            real_data = np.pad(real_data, ((0, 0), (0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
        data_shape = real_data.shape
        real_data_1 = np.random.uniform(0,1,size=data_shape)
        real_data_2 = real_data-real_data_1
        # 计算D前向传播值
        output_d_real_1, output_d_real_2 = netD.forward(real_data_1, real_data_2)
        print('inference: ', (output_d_real_1+output_d_real_2).reshape(-1))
        break
    return 0
    '''

    '''测试SG生成时间'''
    fake_img_1, fake_img_2 = netG.forward(fixed_noise_1, fixed_noise_2)# 一次生成64张图
    fake_tensor = torch.tensor(fake_img_1+fake_img_2)
    print('fake_img: \n',fake_tensor[0])
    # img_list.append(vutils.make_grid(fake_tensor, padding=2, normalize=True))
    """绘图：记录G输出"""
    real_batch = next(iter(cifar_train_loader))
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=False).cpu(),(1,2,0)))
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake_tensor, padding=2, normalize=True).cpu(),(1,2,0)))

    plt.show() 
    return 0

    # 保存图片
    # plt.savefig('./experiment_img/gan_generate/Real_Generate-cifar-Adam-'+str(num_epochs_pre)+'.png')
    
    print("----------start training loop----------")

    for epoch in range(num_epochs):
        # dataloader获取真实图像
        for t, (data, target) in enumerate(cifar_train_loader, 0):
            '''
                (1)先更新D Update D network: minimize -[ log(D(x)) + log(1 - D(G(z))) ]
                训练D的目标是让D更加有能力判断真假数据
            '''
            ## 使用真实数据X进行训练（计算log(D(x))）
            netD.zero_grad() # 训练更新前需要在每个batch中将梯度设置为0
            real_data = data.detach().numpy()
            # print('real_data shape: \n',real_data[0].shape)
            # break
            ## MNIST 数据需要先从1x28x28填充到1x32x32
            # if is_mnist:
            #     real_data = np.pad(real_data, ((0, 0), (0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
            b_size = real_data.shape[0]
            label = np.full((b_size,), real_label)
            # 计算D前向传播值
            output_d_real = netD.forward(real_data).reshape(-1)
            # 计算D真实数据交叉熵损失
            errD_real = loss.forward(output_d_real, label)
            # 计算D的梯度
            dy_errD_real = loss.gradient()
            netD.backward(dy_errD_real)
            
            ## 使用生成数据进行训练（计算log(1 - D(G(z)))）
            noise = np.random.normal(0.0, 1.2, size=(b_size, nz, 1,1)) # 训练每次单独生成噪声
            # G生成假数据
            fake_data = netG.forward(noise)
            label.fill(fake_label)
            # D识别假数据
            output_d_fake = netD.forward(fake_data).reshape(-1)
            # 计算D假数据交叉熵损失
            errD_fake = loss.forward(output_d_fake, label)
            # 计算D的梯度
            dy_errD_fake = loss.gradient()
            netD.backward(dy_errD_fake)

            # 计算总损失
            errD = errD_real+errD_fake

            # 计算D(x),D(G(z))的均值
            D_x = np.mean(output_d_real)
            D_G_z1 = np.mean(output_d_fake)

            # 更新D参数
            optimizerD.step()

            '''
                (2)更新G Update G network: minimize -log(D(G(z)))
            '''
            netG.zero_grad()
            # 填充真实标签，使得交叉熵函数可以只计算log(D(G(z))部分
            label.fill(real_label)
            output_d_fake = netD.forward(fake_data).reshape(-1)
            errG = loss.forward(output_d_fake, label)
            # 计算G的梯度（梯度需要从D传向G）
            dy_errG = loss.gradient()
            dy_netD = netD.backward(dy_errG)
            netG.backward(dy_netD)
            # 计算D(G(z))的均值
            D_G_z2 = np.mean(output_d_fake)
            # 更新G参数（不会去计算D的梯度2333）
            optimizerG.step()

            """输出训练状态"""
            # Loss_D
            # Loss_G
            # D(x)：训练中D对真实数据的平均预测输出
            # D(G(z))：训练中D对虚假数据的平均预测输出（为啥是除法？？）
            if t % 10 == 0:
                end_time1 = time.time()
                print('[%d/%d][%d/%d]\t Loss_D: %.4f\t Loss_G: %.4f\t D(x): %.4f\t D(G(z)): %.4f / %.4f\t train time: %.4f min'
                  % (epoch, num_epochs, t, len(cifar_train_loader), errD, errG, D_x, D_G_z1, D_G_z2, (end_time1-start_time)/60))

            # 记录损失的历史，可以用作画图
            G_losses.append(errG)
            D_losses.append(errD)

            # 记录G生成的图像
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (t == len(cifar_train_loader)-1)):
                fake_img = netG.forward(fixed_noise)# 一次生成64张图
                fake_tensor = torch.tensor(fake_img)
                img_list.append(vutils.make_grid(fake_tensor, padding=2, normalize=True))
            
            iters += 1
        """绘图：记录损失"""
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        # 保存图片
        time_stemp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        plt.savefig('./experiment_img/gan_generate/Loss_fig-cifar-Adam-'+str(epoch+num_epochs_pre)+'('+time_stemp+').png')
        # plt.show()

        """绘图：记录G输出"""
        real_batch = next(iter(cifar_train_loader))
        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))

        # 保存图片
        plt.savefig('./experiment_img/gan_generate/Real_Generate-cifar-Adam-'+str(epoch+num_epochs_pre)+'('+time_stemp+').png')
        # plt.show()
        end_time = time.time()
        print('training time: \n', (end_time-start_time)/60)

        '''存储模型'''
        checkpoint_path = "./model_save/DCGAN_numpy_parameters-cifar-Adam-"+str(epoch+num_epochs_pre)+".pkl"
        torch.save({'epoch':num_epochs+num_epochs_pre, 'D_state_dict':netD.state_dict(), 'G_state_dict':netG.state_dict(), 'G_losses':G_losses, 'D_losses':D_losses}, checkpoint_path)


if __name__ == "__main__":
    test_dcgan()
