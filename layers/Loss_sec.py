'''
Loss_sec.py用于实现损失函数
主要实现交叉熵Cross Entropy
'''
import numpy as np
from Module import Module

class NLLLoss_sec(Module):
    # 计算多分类任务中的负对数似然损失函数，传入logsoftmax([p1,p2,...,pk])
    def __init__(self, size_average=True, bit_length=32):
        super(NLLLoss_sec, self).__init__()
        self.size_average = size_average
        self.bit_length = bit_length

    # 计算损失函数，先计算softmax值，再使用cross entropy计算损失
    def cal_loss(self, prediction_1, prediction_2, labels_1, labels_2):
        '''
            predict：output of predicted probability [batch, [p1,p2,...,pk]]
            labels: labels of dataset [batch, 1] , example: [2,3,8,9]表示类别
            labels可能有one-hot编码模式，[0,0,1,0]代表3
            size_average：if the loss need to be averaged
        '''
        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.prediction_1 = prediction_1
        self.prediction_2 = prediction_2
        self.batchsize = self.prediction_1.shape[0]
        self.loss_1 = 0
        self.loss_2 = 0
        # 判断是否使用one-hot编码
        # if labels_1.ndim >1: # one-hot [[p1,p2,...,pk],...]
        #     for i in range(self.batchsize):
        #         self.loss -= np.sum(self.prediction * self.labels)
        if labels_1.ndim == 1: # [class_num]
            for i in range(self.batchsize):
                # self.loss -= prediction[i, labels[i]]
                self.loss_1 -= self.prediction_1[i, self.labels_1[i]+self.labels_2[i]]
                self.loss_2 -= self.prediction_2[i, self.labels_1[i]+self.labels_2[i]]
        # 对所有样本的loss求平均，作为最终的loss输出
        if self.size_average:
            # self.loss = self.loss/self.batchsize
            self.loss_1 /= self.batchsize
            self.loss_2 /= self.batchsize
        # return self.loss
        return self.loss_1, self.loss_2

    def gradient(self):
        # self.eta = self.labels.copy()
        # 求导结果为-yi
        # self.eta_next = -self.eta
        self.eta_next_1 = -self.labels_1.copy()
        self.eta_next_2 = -self.labels_2.copy()
        # return self.eta_next
        return self.eta_next_1, self.eta_next_2


class BECLoss(Module):
    # 计算二分类任务的交叉熵损失函数
    def __init__(self, size_average=True):
        super(BECLoss, self).__init__()
        self.size_average = size_average
    
    def forward(self, prediction, labels):
        self.prediction = prediction
        self.batchsize = self.prediction.shape[0]
        self.labels = labels

        self.loss = self.labels*np.log(self.prediction)+(1-self.labels)*np.log(1-self.prediction)

        self.loss = -np.sum(self.loss)
        
        if self.size_average:
            self.loss /= self.batchsize
        
        return self.loss

    def gradient(self):
        return self.labels


def test_NLLLoss():
    print('-----NLLLoss test-----')

if __name__ == '__main__':
    test_NLLLoss()