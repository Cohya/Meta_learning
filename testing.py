
from utils import CNN_kerasCfar10
from tasksGenerator import TasksGeneratorCfar10
import numpy as np 
import tensorflow as tf 
loder = TasksGeneratorCfar10()

task = loder.all_combinatoin[0]

tf.debugging.set_log_device_placement(False)
X_train, Y_train, X_test, Y_test = loder.load_specific_task(0)

X_train = np.float32(X_train)
X_test = np.float32(X_test)

Y_test = np.int32(Y_test)
Y_train = np.int32(Y_train)

K = 4
net = CNN_kerasCfar10(4)

net.train(X_train, Y_train, X_test, Y_test, 32, 100)
# 

# inputs = tf.keras.Input(shape = (32,32,3) )
# x = tf.keras.layers.Conv2D(filters = 32, 
#                            kernel_size=(3,3),
#                            activation= 'relu',
#                            )(inputs)
# # x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation= 'relu')(x)
# x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
# # x = tf.keras.layers.Dropout(0.25)(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# # x = tf.keras.layers.Dropout(0.25)(x)
# outputs = tf.keras.layers.Dense(K, activation='softmax')(x)

# model = tf.keras.Model(inputs=inputs, outputs=outputs)

# # self.trainable_params = self.model.trainable_variables 

# model.compile(optimizer='adam', 
#               loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# model.fit(X_train, Y_train ,epochs=100, batch_size=64) 
