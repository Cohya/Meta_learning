import numpy as np 
import tensorflow as tf 
from utils import load_spesific_task, all_combination, CNN_keras
from Meta_Learning import FOMAML
import pickle 
# import sys 

# sys.stdout = open("test.txt", "w")
# First lat's prepare all the tasks 
# all_combs = all_combination([0,1,2,3,4,5,6], 3)  

# tasks = []

# for classes in all_combs:
#     # each task has the structure of (X_train, Y_train, X_test, Y_test) # normelized
#     task = load_spesific_task(classes)
#     tasks.append(task)
    
# with open('data.pk', 'wb') as file:
#     pickle.dump(tasks, file)5

with open('data.pk', 'rb') as file:
    tasks = pickle.load(file)
    
net = CNN_keras()
fomaml = FOMAML(net = net, tasks = tasks, optimizers = [None] * len(tasks),
                loss_fucntions = [tf.keras.losses.CategoricalCrossentropy()] * len(tasks),
                k = 1)


fomaml.train(iterations = 1)

# sys.stdout.close()