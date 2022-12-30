import numpy as np 
import tensorflow as tf 
from utils import load_spesific_task_cfar10, load_spesific_task_mnist, all_combination, CNN_keras
from Meta_Learning import FOMAML
import pickle 
from sklearn.utils import shuffle
# import sys 
import matplotlib.pyplot as plt 
from tasksGenerator import TasksGeneratorCfar10
import matplotlib
# sys.stdout = open("test.txt", "w")
# First lat's prepare all the tasks
K = 3
all_combs = all_combination([0,1,2,3,4,5], K)  
problem = 'cfar10'
tasks = []

# for classes in all_combs:
#     # each task has the structure of (X_train, Y_train, X_test, Y_test) # normelized
#     if problem =='MNIST':
#         task = load_spesific_task_mnist(classes)# Mnist
#     elif problem == 'cfar10':
#         task = load_spesific_task_cfar10(classes)
    
#     tasks.append(task)
    
    
# with open('fashion_data.pk', 'wb') as file:
#     pickle.dump(tasks, file)

# with open('data.pk', 'rb') as file:
#     tasks = pickle.load(file)
    
# with open('fashion_data.pk', 'rb') as file:
#     tasks = pickle.load(file)
    
if problem == 'MNIST':
    shape = (28,28,1)
    cmap = 1
elif problem == 'cfar10':
    shape = (32, 32, 3)
    cmap = 3
    
# cmap =  1
tasksGenerator = TasksGeneratorCfar10()
net = CNN_keras(K = K, shape = shape, cmap= cmap )
fomaml = FOMAML(net = net,
                tasksGenerator=tasksGenerator)


best_init_weights = fomaml.train(iterations = 10000,
                                  optimizers = [None] * tasksGenerator.n_possible_tasks, 
                                  loss_fucntions = [tf.keras.losses.CategoricalCrossentropy()] * tasksGenerator.n_possible_tasks,
                                  )

with open('weights_after_fomaml_cdar10.pk', 'wb') as file:
    pickle.dump(best_init_weights, file)

# # with open('weights_after_fomaml.pk', 'rb') as file:
# #     best_init_weights= pickle.load(file)
# # # # sys.stdout.close()

### Train from scratch 
acc_test = []
acc_train = []

acc_fomaml_train =[]
acc_fomaml_test = []

loss_train_matrix = []
loss_test_matrix = []
loss_train_fomaml_matrix = []
loss_test_fomaml_matrix = []
for i in range(3):
    if problem =='MNIST':
        task = load_spesific_task_mnist([7,8,9])# Mnist
    elif problem == 'cfar10':
        task = load_spesific_task_cfar10([7,8,9])

    X_train, Y_train, X_test, Y_test = task
    
    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)
    net = CNN_keras(K = K, shape = shape, cmap = cmap)
    
    
    loss_train_vec, loss_test_vec= net.train(X_train = X_train[:32],
              Y_train = Y_train[:32],
              X_test = X_test[:32], 
              Y_test = Y_test[:32],
              batch_sz = 10,
              epochs = 9) # was 128 for the test 
    
    loss_train_matrix.append(loss_train_vec)
    loss_test_matrix.append(loss_test_vec)
    
    acc_test.append(net.accuracy(X_test, Y_test))
    acc_train.append(net.accuracy(X_train[:20], Y_train[:20]))
    ### TRain with best weights 
    

    net2 = CNN_keras(K = K, shape = shape, cmap = cmap)
    net2.load_weights(best_init_weights)
    
    
    loss_train_vec_fomaml, loss_test_vec_fomaml = net2.train(X_train = X_train[:32],
              Y_train = Y_train[:32],
              X_test = X_test[:32], 
              Y_test = Y_test[:32],
              batch_sz = 10,
              epochs = 9)
    
    loss_train_fomaml_matrix.append(loss_train_vec_fomaml)
    loss_test_fomaml_matrix.append(loss_test_vec_fomaml)
    
    
    
    acc_fomaml_test.append(net2.accuracy(X_test, Y_test))
    acc_fomaml_train.append(net2.accuracy(X_train[:20], Y_train[:20]))
    
    
# plt.figure(20)
# plt.plot(np.arange(len(acc_fomaml_train)), acc_fomaml_train)

# plt.figure(2)
# plt.plot(np.arange(len(acc_train)), acc_train)


# ### create the mean and teh std 
loss_train_fomaml_matrix = np.array(loss_train_fomaml_matrix)
loss_test_fomaml_matrix = np.array(loss_test_fomaml_matrix)

loss_train_matrix = np.array(loss_train_matrix)
loss_test_matrix = np.array(loss_test_matrix)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.labelsize']= 12
plt.rcParams['ytick.labelsize']= 12

fig, ax  = plt.subplots()
N = len(loss_test_vec)

mean_train = np.mean(loss_train_matrix, axis = 0)
std_train = np.std(loss_train_matrix, axis = 0)
ax.plot(np.arange(N), mean_train, c = 'r',label = 'Train -Random Initialization')
ax.fill_between(np.arange(N),mean_train - std_train, std_train+mean_train, alpha=0.3,facecolor = 'r' )




mean_train_fomaml = np.mean(loss_train_fomaml_matrix, axis = 0)
std_train_fomaml = np.std(loss_train_fomaml_matrix, axis = 0)
ax.plot(np.arange(N),mean_train_fomaml, label = 'Train - FOMAML', color = 'g')
ax.fill_between(np.arange(N),mean_train_fomaml - std_train_fomaml,
                mean_train_fomaml + std_train_fomaml, alpha=0.3, facecolor = 'g')


mean_test = np.mean(loss_test_matrix, axis = 0)
std_test = np.std(loss_test_matrix, axis = 0)
plt.plot(np.arange(N), mean_test, label = 'Test -Random Initialization', color = 'b')
plt.fill_between(np.arange(N), -std_test+mean_test, std_test+mean_test, alpha=0.3, facecolor = 'b')


mean_test_fomaml = np.mean(loss_test_fomaml_matrix, axis = 0)
std_test_fomaml = np.std(loss_test_fomaml_matrix, axis = 0)
ax.plot(np.arange(N), mean_test_fomaml , label = 'Test - FOMAML', color = 'k')
ax.fill_between(np.arange(N),mean_test_fomaml - std_test_fomaml,
                mean_test_fomaml + std_test_fomaml, alpha=0.3, facecolor = 'k')



plt.legend(frameon = False)


plt.xlabel('Epoch')
plt.ylabel('Loss-Value')

# plt.savefig('fromData_0345SGD.png')