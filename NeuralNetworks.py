import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)

# 生成数据
def generate_data(num_sample = 1000):
    np.random.seed(0)
    X, y = datasets.make_moons(num_sample, noise=0.20)
    return X, y

# 可视化数据
def visualize(x,y,output):
    rounded_preds = np.round(output)
    yy = np.zeros((len(y),1))
    for i in range(len(y)):
        if y[i] == rounded_preds[i]:
            if y[i] == 1:
                yy[i] = 1
            else:
                yy[i] = 0
        else:
            if y[i] == 1:
                yy[i] = 2  # 把0错误预测1
            else:
                yy[i] = 3  # 把1错误预测0
    # 作散点图
    plt.scatter(x[:, 0], x[:, 1], s=40, c=yy, cmap=plt.cm.Spectral)

# 定义神经网络
class NeuralNetwork():
    def __init__(self, input_size = 10, hidden_size = 20, learn_rate = 0.1):
        # 输入神经元个数
        self.input_size = input_size
        # 隐藏层个数
        self.hidden_size = hidden_size
        # 样本数量
        self.num_samples = 0
        # 初始化参数
        self.w1 = np.random.normal(loc=0.0, scale=1.0, size=(self.input_size, self.hidden_size))
        self.w2 = np.random.normal(loc=0.0, scale=1.0, size=(self.hidden_size, 1))
        self.b1 = np.random.normal(loc=0.0, scale=1.0, size=(1,self.hidden_size))
        self.b2 = np.random.normal(loc=0.0, scale=1.0, size=(1,1))
        self.z1,self.z2,self.a1,self.a2 = 0,0,0,0
        # 学习率
        self.learn_rate = learn_rate

    # 输出层激活函数sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid的求导
    def sigmoid_derivative(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    # 神经元激活函数tanh
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # tanh求导
    def tanh_derivative(self, x):
        fx = self.tanh(x)
        return 1 - fx**2

    # 损失函数
    def Loss(self, y, output):
        return ((y - output) ** 2).mean()

    # 向前传播
    def forward(self,input):
        # 第一层网络 input: [num_samples, input_size]
        # z1: [num_samples,hidden_size]
        self.z1 = np.dot(input,self.w1) + self.b1
        # a1: [num_samples,hidden_size]
        self.a1 = self.tanh(self.z1)
        # 第二层网络
        # z2: [num_sample, 1]
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        # a2: [num_sample, 1]
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    # 反向传播
    def backprod(self, y, a2, input):
        self.num_samples = len(y)
        # 上游误差 delta3:[num_sample, 1]
        delta3 = -(y - a2)*(self.sigmoid_derivative(self.z2))
        delta3 = delta3.reshape(self.num_samples, 1)
        # w2 的梯度 [hidden_szie, 1]
        dw2 = (self.a1.T).dot(delta3)/self.num_samples
        # b2 的梯度 [1,1]
        db2 = sum(delta3)/self.num_samples
        # 上游误差 delta2:[num_samples, hidden_size]
        delta2 = delta3*(self.w2.T)*self.tanh_derivative(self.z1)
        # w1 的梯度 [input_size, hidden_size]
        dw1 = (input.T).dot(delta2)/self.num_samples
        # b1 的梯度 [1,hidden_size]
        db1 = sum(delta2)/self.num_samples
        # 权重更新
        self.w2 = self.w2 - self.learn_rate * dw2
        self.b2 = self.b2 - self.learn_rate * db2
        self.w1 = self.w1 - self.learn_rate * dw1
        self.b1 = self.b1 - self.learn_rate * db1

# 准确率
def acc(y,output):
    rounded_preds = np.round(output)
    corrects = sum(rounded_preds == y)
    a = corrects/len(y)
    return a

epoch = 500
# 样本数量
num_samples = 200
# 生成数据
x,y = generate_data(num_samples)
y = y.reshape(num_samples,1)
# 定义神经网络
model = NeuralNetwork(input_size = 2, hidden_size = 6)
loss = [] # 保存损失函数
# 进行迭代
for i in range(epoch):
    print('epoch:',i)
    # 向前传播
    output = model.forward(x)
    # 计算损失函数
    loss.append(model.Loss(y, output))
    # 反向传播
    model.backprod(y, output, x)
    # 训练集合准确率
    accuracy = acc(y, output)
    print('准确率:',accuracy)

# 可视化数据
visualize(x,y,output)
xx = np.linspace(1,epoch,epoch)
plt.plot(xx,loss,'r',label='loss')
