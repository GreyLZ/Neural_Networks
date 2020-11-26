import numpy as np
from sklearn import datasets

# 神经网络
class NN():
    def __init__(self):
        self.learn_rate = 0.2         # 学习率
        self.input_len = 2
        self.hidden_len = 4
        self.num_samples = 0
        # 初始化参数
        self.w1 = np.random.randn(self.input_len, self.hidden_len)
        self.w2 = np.random.randn(self.hidden_len, 1)
        self.b1 = np.random.randn(1,self.hidden_len)
        self.b2 = np.random.randn(1,1)
        self.z1,self.z2,self.a1,self.output = 0,0,0,0


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidderiv(self, x):
        f = self.sigmoid(x)
        return f * (1 - f)

    def forward(self,input):
        # 第一层网络
        self.z1 = np.dot(input,self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        # 第二层网络
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.output = self.sigmoid(self.z2)
        return self.output

    # 反向传播
    def backward(self, y, a2, input):
        self.num_samples = len(y)
        delta3 = -(y - a2)*(self.sigmoidderiv(self.z2))
        delta3 = delta3.reshape(self.num_samples, 1)
        dw2 = (self.a1.T).dot(delta3)/self.num_samples
        db2 = sum(delta3)/self.num_samples
        delta2 = delta3*(self.w2.T)*self.sigmoidderiv(self.z1)
        dw1 = (input.T).dot(delta2)/self.num_samples
        db1 = sum(delta2)/self.num_samples
        # 权值更新
        self.w2 = self.w2 - self.learn_rate * dw2
        self.b2 = self.b2 - self.learn_rate * db2
        self.w1 = self.w1 - self.learn_rate * dw1
        self.b1 = self.b1 - self.learn_rate * db1

# 准确率
def acc(y,output):
    rounded_preds = np.round(output)
    accuray = sum(rounded_preds == y)/len(y)
    return accuray

# 数据
np.random.seed(0)
x, y = datasets.make_moons(200, noise=0.22)
y = y.reshape(200,1)
nn = NN()
for i in range(500):
    print('epoch:',i)
    output = nn.forward(x)
    nn.backward(y, output, x)
    # 准确率
    accuracy = acc(y, output)
    print('准确率:',accuracy)
