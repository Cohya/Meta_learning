
import numpy as np 
from utils import load_spesific_task, LinearNet
from Meta_Learning import FOMAML
import tensorflow as tf 

"""
Here we are going to test first order MAML over MNIST data !!!
"""

tasks = [] 

# X_train, Y_train, X_test, Y_test = task 

tasks_classes = []

X = np.random.randn(10,2).astype(np.float32)
Y =  np.expand_dims(np.float32(np.sum(X, axis = 1) + np.random.randn(10)), axis = 1)

X_train1 = X[0:6, :]
Y_train1 = Y[:6]

X_test1 = X[6:,...]
Y_test1 = Y[6:,...]

X2 = np.random.randn(10,2).astype(np.float32)
Y2 =  np.expand_dims(np.float32(np.sum(2*X2, axis = 1) + np.random.randn(10)), axis = 1)

X_train2 = X2[0:6, :]
Y_train2 = Y2[:6]

X_test2 = X2[6:,...]
Y_test2 = Y2[6:,...]


#### Task 1 
task1  = [X_train1, Y_train1, X_test1, Y_test1] 
task2 = [X_train2, Y_train2, X_test2, Y_test2]
########################
net = LinearNet(2, 1)
loss_fucntions = tf.keras.losses.mean_squared_error

fomaml = FOMAML(net = net, tasks= [task1, task2],
                optimizers = [0]*2,
                loss_fucntions = [tf.keras.losses.mean_squared_error]*2,
                k = 2)

############3 calculate phi

phi_1 = fomaml.calc_phi(X_train1, Y_train1, None, loss_fucntions)
phi_2 = fomaml.calc_phi(X_train2, Y_train2, None, loss_fucntions)
## Analytical grads 
W = net.trainable_params[0].numpy()
a = np.matmul(X_train1.T, X_train1)
grads = (np.matmul(a,W)
         - np.matmul(X_train1.T, Y_train1)) * 2


W = net.trainable_params[0].numpy()
a = np.matmul(X_train2.T, X_train2)
grads2 = (np.matmul(a,W)
         - np.matmul(X_train2.T, Y_train2)) * 2
print(grads)

phi_k_anlytic1 = W - 0.1*grads
phi_k_anlytic2 = W - 0.1*grads2

print("tf calc:", phi_1[0].numpy())
print("Analytical calc:", phi_k_anlytic1)  

##################333 update theta 
## Analytic vs tensorflow 
### Done over the test data
### The test 1 
I  = np.eye(2)
a_1 = (I - 2 * 0.1 * np.matmul(X_test1.T, X_test1))
b1 = 2 * (np.matmul(np.matmul(X_test1.T, X_test1), phi_k_anlytic1) - np.matmul(X_test1.T, Y_test1))
res_1 = np.matmul(a_1, b1)


a_2 = (I - 2 * 0.1 * np.matmul(X_test2.T, X_test2))
b2 = 2 * (np.matmul(np.matmul(X_test2.T, X_test2), phi_k_anlytic2) - np.matmul(X_test2.T, Y_test2))
res_2 = np.matmul(a_2, b2)


res_11 = b1
res = res_1 + res_2 
theta_new = W - 0.01 * res

new_theta = fomaml.train(iterations = 1)

print()
print("New theta: based on anlytic calculation:", new_theta)

print()
print("New theta based on tf:", fomaml.net.trainable_params[0])