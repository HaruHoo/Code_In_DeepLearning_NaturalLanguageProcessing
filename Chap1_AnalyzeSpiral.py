import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import SGD
from dataset import spiral
from ch01.two_layer_net import TwoLayerNet

#设置超参数
max_epoch=300
batch_size=30
hidden_size=10
learning_rate=1.0

#读入数据
x,t=spiral.load_data()
model=TwoLayerNet(input_size=2,hidden_size=hidden_size,output_size=3)
optimizer=SGD(lr=learning_rate)

#学习用的变量
data_size=len(x)#输入数据的长度
max_iters=data_size// batch_size  #//表示整数除法,它可以返回商的整数部分(向下取整)
total_loss=0
loss_count=0
loss_list=[]

for epoch in range(max_epoch):#跑max_epoch次
    #打乱数据
    idx=np.random.permutation(data_size)
    x=x[idx]
    t=t[idx]

    for iters in range(max_iters):
        batch_x = x[iters * batch_size:(iters + 1) * batch_size]
        batch_t = t[iters * batch_size:(iters + 1) * batch_size]

        #计算梯度，更新参数
        loss=model.forward(batch_x,batch_t)#前向传播得loss
        model.backward()#反向传播计算梯度，优化参数
        optimizer.update(model.params,model.grads)

        total_loss+=loss
        loss_count +=1

        #定期输出学习过程
        if(iters+1)%10==0:
            avg_loss=total_loss/loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f'
                  %(epoch+1,iters+1,max_iters,avg_loss))
            loss_list.append(avg_loss)
            total_loss,loss_count=0,0

# 绘制学习结果
plt.plot(np.arange(len(loss_list)), loss_list, label='train')  #np.arrange(),参数为1表示默认起点为0，终点为参数，步长为1
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

# 绘制决策边界
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 绘制数据点
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()
