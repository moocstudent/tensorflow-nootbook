import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

now_begin = datetime.datetime.now().microsecond
# 线性回归
train_X = np.linspace(-1,1,100)

# np.tandom.randn(*train_X.shape) 等价于 np.random.randn(100)
train_Y = 2* train_X+np.random.randn(*train_X.shape)*0.3

plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()
now_end = datetime.datetime.now().microsecond
print(now_end-now_begin)