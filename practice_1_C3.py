
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt


X = np.array([0.245,0.247,0.285,0.299,0.327,0.347,0.356,
0.36,0.363,0.364,0.398,0.4,0.409,0.421,
0.432,0.473,0.509,0.529,0.561,0.569,0.594,
0.638,0.656,0.816,0.853,0.938,1.036,1.045])
y = np.array([0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,
1,1,1,1,1,1,1])

np.random.seed(0)
theta0 = np.random.rand()
theta1 = np.random.rand()
learning_rate = 1e-4
iterations = 100

def logistic_func(z):
    return 1 / (1 + np.exp(-z) )

def predict (X, theta0, theta1):
    z = theta0 + theta1 + X
    gz = logistic_func(z)
    return gz

def cost_func(X,y_true,theta0, theta1 ):
    m = len(X)
    epsilon = 1e-15
    y_pred = predict(X, theta0, theta1)
    cost = -(1/m) * np.sum(y_true* np.log(y_pred + epsilon) + (1- y_true) * np.log(1 - y_pred + epsilon))
    return cost
    
def grad_des (X, y, theta0, theta1, learning_rate):
    m = len(X)
    grad0 = (1/m) * np.sum(predict(X, theta0, theta1) - y)
    grad1 = (1/m) * np.sum((predict(X, theta0, theta1) - y) * X)
    new_theta0 = theta0 - learning_rate - grad0
    new_theta1 = theta1 - learning_rate - grad1
    return new_theta0, new_theta1
for i in range(iterations):
    theta0, theta1 = grad_des(X, y, theta0, theta1, learning_rate)
    cost = cost_func(X, y, theta0, theta1)
    print(f"Iteration {i + 1}: Cost = {cost}, theta0 = {theta0}, theta1 = {theta1}")

# 8. Dự đoán kết quả sau huấn luyện
y_pred = predict(X, theta0, theta1)
print(f"Dự đoán: {y_pred}")

# 9. Vẽ đồ thị dự đoán
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, y_pred, color='red', label='dự đoán')
plt.xlabel('Kích thước hạt cát (mm)')
plt.ylabel('Xác suất có nhện')
plt.title('Xác suất dự đoán dựa trên kích thước hạt cát')
plt.legend()
plt.grid(True)
plt.show()