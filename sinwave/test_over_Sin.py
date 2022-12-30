

# from utils import ANN_regressionSin
import tensorflow as tf 
import numpy as np 
from nets_not_keras import ANN
from tasksGenerator import taskGeneratorSin
from Meta_Learning_2order import MAML
import pickle
import matplotlib.pyplot as plt 

generator = taskGeneratorSin()
task = generator.get_task(32)
# net = ANN_regressionSin()
net = ANN(1, 1, [40,40], trainable=True)
net_copy = ANN(1,1,[40,40], trainable=False)

tasks,_,_= generator.sample_batch_of_tasks(8, 150)


task = tasks[0]

# with open('task.pk', 'wb') as file:
#     pickle.dump(task, file)
    
    
with open('task.pk', 'rb') as file:
    task = pickle.load(file)
    
    
X_train , y_train, X_test, y_test = task

# fomaml = MAML(net = net,
#                 tasksGenerator=generator,
#                 copy_net=net_copy)


# best_init_weights = fomaml.train(iterations = 10000,
#                                   optimizers = [None] * 7, 
#                                   loss_fucntion = tf.keras.losses.MSE ,
#                                   samples_per_class = 10)

# with open('weights_for_sin.pk', 'wb') as file:
#     pickle.dump(best_init_weights, file)

with open('weights_for_sin.pk', 'rb') as file:
    best_init_weights = pickle.load(file) 
    
net.copy_to_net_weights(best_init_weights)
epochs = 15
samples = 10
loss_MAML,loss_MAML_test = net.train(X = X_train[:samples], Y = y_train[:samples],
                                     X_test=X_train[samples:], Y_test=y_train[samples:],
                                     epochs=epochs)

y_hat = net(X_train)


net2 = ANN(1, 1, [40,40], trainable=True)
loss_rand,loss_rand_test = net2.train(X = X_train[:samples], Y = y_train[:samples],
                                      X_test=X_train[samples:], Y_test=y_train[samples:],
                                      epochs= epochs)
y_hat2 = net2(X_train)

y_hat = np.squeeze(y_hat.numpy())
X_train = np.squeeze(X_train)

X_train0 = np.array(X_train)
X_train1 = np.array(X_train)
X_train2 = np.array(X_train)

a = sorted(zip(X_train1, y_hat), key=lambda x: x[0], reverse=False)
X_train1, y_hat = zip(*a)
y_train = np.squeeze(y_train)
y_train0 = np.array(y_train)

a = sorted(zip(X_train2, y_hat2), key=lambda x: x[0], reverse=False)
X_train2, y_hat2 = zip(*a)



a = sorted(zip(X_train0, y_train0), key=lambda x: x[0], reverse=False)
X_train0, y_train0 = zip(*a)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

plt.plot(X_train1,y_hat, label = 'MAML')
plt.plot(X_train0,y_train0, label = 'Ground truth')
plt.plot(X_train2, y_hat2, label = 'random')

plt.scatter(X_train[:samples], y_train[:samples],marker = '*', c = 'r', label = 'Training Data')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(frameon=False)



plt.figure(2)
plt.plot(loss_MAML, 'b', label = 'MAML - train')
plt.plot(loss_rand, 'g', label ='rand - train')

plt.plot(loss_MAML_test, '--b', label= 'MAML - test')
plt.plot(loss_rand_test, '--g', label ='rand - test')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(frameon = False)
