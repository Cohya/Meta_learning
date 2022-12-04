import numpy as np 
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt 
import tensorflow as tf 


def one_hot_encode(Y, classes):
    n = len(Y)
    Y_encode = np.zeros(shape = (n, classes))
    
    for i in range(n):
        Y_encode[i][Y[i]] = 1
    
    return np.int32(Y_encode)
    
    
def load_mnist_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    # normelized data 
    X_train = np.float32(X_train / 255.0)
    X_test = np.float32(X_test / 255.0)
    
    # One_hot encoding 
    classes = max(len(np.unique(Y_test)), len(np.unique(Y_train)))
    
    Y_train_encode = one_hot_encode(Y_train, classes)
    Y_test_encode = one_hot_encode(Y_test, classes)
    
    return X_train, Y_train_encode, X_test, Y_test_encode, Y_train, Y_test



def creat_one_hot(class_i, n, number_of_classses):
    Y = np.zeros(shape = (n, number_of_classses))
    for i in range(n):
        Y[i][class_i] = 1
        
    return Y 
        
def load_task_mnist(classes, X_train_original, 
                    Y_train_original,
                    X_test_original,
                    Y_test_original):
    """
    Parameters
    ----------
    classes : list of classes for the test 

    Returns: Data of the mnist classes for the specipic classes
    -------
    None.

    """
    number_of_classses = len(classes)
    
    
    for i in range(len(classes)):
        c = classes[i]
        X_test_c = X_test_original[Y_test_original == c]
        X_train_c = X_train_original[Y_train_original == c]
        
        Y_test_c = creat_one_hot(i, len(X_test_c), number_of_classses)
        Y_train_c = creat_one_hot(i, len(X_train_c), number_of_classses)
        
        if i == 0 :
            X_train = np.float32(X_train_c)
            X_test = np.float32(X_test_c)
            Y_train = np.int32(Y_train_c)
            Y_test = np.int32(Y_test_c)
        else:
            X_train = np.concatenate((X_train, X_train_c), axis = 0)
            Y_train = np.concatenate((Y_train, Y_train_c), axis = 0)
            X_test = np.concatenate((X_test, X_test_c), axis = 0)
            Y_test = np.concatenate((Y_test, Y_test_c), axis = 0)
            
    
    return X_train, Y_train, X_test, Y_test
        
def show_number_if_vector(X):
    n = len(X)
    r = np.square(n)
    c = np.square(n)
    image = np.zeros(shape = (r,c))
    
    for i in range(n):
        image[i//n][i%n] = X[i]
        
    plt.imshow(image)
    
def show_number(X):
    plt.imshow(X, cmap='gray')
    
def load_spesific_task(classes):   
    # classes = list of class number 
    X_train, Y_train_encode, X_test, Y_test_encode, Y_train, Y_test =  load_mnist_data()
       
    X_train, Y_train, X_test, Y_test = load_task_mnist(classes,
                                                       X_train,
                                                       Y_train,
                                                       X_test,
                                                       Y_test) 
    
    
    return X_train, Y_train, X_test, Y_test 


    
class LinearNet():
    
    def __init__(self, X_dims, output):
        self.W = tf.Variable(
                            initial_value = tf.random.normal(shape = (X_dims, output), stddev = 0.002),
                            name =( "W_" + str(0)))
        
        
        self.trainable_params = [self.W]
    def __call__(self, X, train):
        return tf.matmul(X, self.W)
    


def all_combination(vec, set_size):
    def all_combinatio_helper(vec, subSet, n, m):
        if (n+m) < set_size:
            return []
        
        elif m == 3:
            return subSet
        
        # print(subSet, vec[0])
        subSet1 = list(subSet)
        subSet1.append(vec[0])

        # print(subSet1)
        res1 = all_combinatio_helper(vec[1:], subSet = subSet1, n = n-1, m = m + 1)
        res2 = all_combinatio_helper(vec[1:], subSet=subSet, n = n - 1, m = m )
        
        
        return res1 + res2 
    n = len(vec)
    all_subSets = all_combinatio_helper(vec, subSet= list([]), n = n, m = 0)
    res = []
    n2 = len(all_subSets)
    for i in range(n2//set_size):
        res.append(all_subSets[i*set_size:(i+1)*set_size])
    
    return res 
    
    
class CNN_keras():
    def __init__(self):
        
        inputs = tf.keras.Input(shape = (28,28,1))
        x = tf.keras.layers.Conv2D(filters = 32, 
                                   kernel_size=(3,3),
                                   activation= 'relu',
                                   )(inputs)
        x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation= 'relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        self.trainable_params = self.model.trainable_variables
        
    def __call__(self,x, train):
        x = np.expand_dims(x, axis = -1)
        print("data shape to net ", x.shape)
        return self.model(x, training = train)
        
class CNN_keras_2():
    def __init__(self):
        
        inputs_shape = (28,28,1)
        layer1 = tf.keras.layers.Conv2D(filters = 32, 
                                   kernel_size=(3,3),
                                   activation= 'relu',
                                   padding='valid'
                                   ) # (26,26,32)
        layer_2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation= 'relu') # (24,24,64)
        layer_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) # (12,12,64)
        layer_4 = tf.keras.layers.Dropout(0.25)
        layer_5 = tf.keras.layers.Flatten()
        layer_6 = tf.keras.layers.Dense(128, activation='relu')
        layer_7 = tf.keras.layers.Dropout(0.25)
        output_layer = tf.keras.layers.Dense(3, activation='softmax')
        
        # self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        ## Initialize the net
        
        #collect trainable_params
        self.trainable_params = self.model.trainable_variables
        
    def __call__(self,x, train):
        x = np.expand_dims(x, axis = -1)
        
        
        
        return        
        
       
    
    
        
                             
        
        
        
    
    