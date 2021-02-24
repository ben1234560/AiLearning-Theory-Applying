import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
data = pd.read_csv('../data/world-happiness-report-2017.csv')  # 导入数据
# 得到训练和测试数据，以8：2切分
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'  # 特征features
output_param_name = 'Happiness.Score'  # 标签label

x_train = train_data[[input_param_name]].values  # 构建数据
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

# 可视化展示 run, 可以看到训练数据和预测数据的分布
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

# 训练线性回归模型
num_iterations = 500  # 迭代次数
learning_rate = 0.01  # 学习率

linear_regression = LinearRegression(x_train, y_train)  # 初始化模型
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('GD')
plt.show()
