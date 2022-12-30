import tensorflow as tf 
from nets_not_keras import ANN

import numpy as np 

net_trainable = ANN(3, 1, [3], trainable = True)
net_trainable2 = ANN(3, 1, [3], trainable = False)
# class dense():
    
#     def __init__(self,trainable = True):
#         self.x = 3.0
#         if trainable:
#             self.x = tf.Variable(1.0)
        
    
#     def forward(self,a):
#         return self.x**2
    
    
# layer1 = dense(trainable = True)
# layer2 = dense(trainable = False)



## Amazing how to take the second derivative 

# a= np.random.normal(size = (1,3)).astype(np.float32)
# with tf.GradientTape() as tape:
    
#     with tf.GradientTape() as tape2:
#         cost = net_trainable(a)
        
#     grads = tape2.gradient(cost, net_trainable.trainable_params)
#     print(grads[0])
#     # counter = 0
#     for counter in  range(len(net_trainable2.trainable_params)):
#         print(net_trainable2.trainable_params[counter])
#         print("$$$$$$$$$$$$$$$$$$$$$$$$$4")
#         net_trainable2.trainable_params[counter] = 0.001 * grads[counter] +  net_trainable.trainable_params[counter]
#         print(net_trainable2.trainable_params[counter])
#         # counter += 1
        
#     net_trainable2.copy_weights()
#     k = net_trainable2(a)
# grads2 = tape.gradient(k, net_trainable.trainable_params)
    



