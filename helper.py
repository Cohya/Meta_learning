
import tensorflow as tf 
import numpy as np 

############333Tking the hessian based on graph computation 
# # hes(x)
# @tf.function
# def get_gradients(inputs):
#     loss = tf.math.square(inputs[0]) + inputs[1]
#     g = tf.gradients(loss, inputs)
#     g2 = tf.gradients(g, inputs)
#     return g, g2


# @tf.function
# def get_hessian(inputs):
#     loss = tf.math.square(inputs)
#     # loss = tf.reduce_sum(model(inputs))
#     return tf.hessians(loss, inputs)

# batch_size = 3
# tf.random.set_seed(123)
# test_input = tf.convert_to_tensor(3.0)#tf.random.uniform((3,10),minval=1.5,maxval=2.5)
# hessian = get_hessian(test_input)
# print(type(hessian))
# print(len(hessian))

# g, g2 = get_gradients([test_input, test_input])
# print(g)
# print( g2)


# # print(hessian[0].shape)
# print(hessian[0][0,0,0,0])
# print(hessian[0][0,0,0,1])

############### Copy a class using inheritance
# class Net():
#     def __init__(self,):
        
#         inp = tf.keras.Input(shape = (1, 10))
#         x = tf.keras.layers.Dense(125)(inp)
#         output = tf.keras.layers.Dense(3)(x)
        
        
#         self.model = tf.keras.Model(inputs = inp, outputs = output)
#         self.trainable_params = self.model.trainable_weights
        
#     def __call__(self,x):
#         return self.model(x)
    
# class CopyNet(Net):
#     pass

# # Inheritance is the right way of copies a class
    
    
# x = tf.random.normal(shape = (3,10))
# model = Net()
# # y = model(x)      
# model_copy = CopyNet()


################ Reading the data 

















