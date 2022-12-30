
from sklearn.utils import shuffle
import tensorflow as tf 
import numpy as np




class DenseLayer(object):
    def __init__(self, M1, M2, activation = tf.nn.relu, use_bias = True, i_d = 0, trainable = True):
        
        self.use_bias = use_bias
        self.f = activation
        # we need to use scaling using 0 and var = 1 ~N(0,1)
        # He normalization 
        W0= tf.convert_to_tensor(np.random.randn(M1, M2).astype(np.float32) * np.sqrt(2./float(M1)))
        if trainable:
            self.W = tf.Variable(initial_value = W0, name = 'W_dense_%i' % i_d)
        else:
            self.W = W0
            
        self.trainable_params = [self.W]
        
        if self.use_bias:
            self.b = tf.Variable(initial_value = tf.zeros([M2,]), 
                                 name = 'b_dense_%i' % i_d)
            
            if trainable == False:
                self.b = tf.zeros([M2,])
            self.trainable_params.append(self.b)
        
    def forward(self, X):
        Z = tf.matmul(X, self.W)
        
        if self.use_bias:
            Z += self.b
            
        return self.f(Z)
    
    
class ANN():
    def __init__(self, input_dims, output_dims, hidden_layer_sizes, last_layer_activation = tf.identity, trainable = True):
        
        """
        input_dims = is the input dimension 
        output_dims = is the output dimension 
        hidden_layer_sizes = is a vector with the hidden layer size e.g. [100,100]
        
        """

        self.layers = []
        
        # Let's build the layers
        M1 = input_dims
        id_counter = 0
        for M2 in hidden_layer_sizes:
            layer = DenseLayer(M1 = M1, M2 = M2, i_d = id_counter, trainable=trainable)
            
            self.layers.append(layer)
            id_counter += 1
            M1 = M2
            
        
        ## last layer 
        
        last_layer = DenseLayer(M1,
                                M2 = output_dims,
                                i_d = id_counter, 
                                activation = last_layer_activation, trainable=trainable)
        
        self.layers.append(last_layer)
        
        ## collect all the trainable params 
        self.trainable_params = []
        
        for layer in self.layers:
            self.trainable_params += layer.trainable_params
            
    def copy_weights(self):
        counter = 0
        for layer in self.layers:
            layer.W = self.trainable_params[counter]
            layer.b = self.trainable_params[counter+1]
        
        counter += 2
        
    def __call__(self,X, train = False):
        Z = X 
        
        for layer in self.layers:
            Z = layer.forward(Z)
        # print(layer.trainable_params[-1]) 
        return Z
        
    def copy_to_net_weights(self,weights):
        for w, wc in zip(self.trainable_params, weights):
            w.assign(wc)
            
            
    def train(self, X, Y, X_test, Y_test, epochs = 5):
        self.opt = tf.keras.optimizers.Adam(0.001)
        n = len(X)
        batch_size = 32
        n_batchs = max(n // batch_size, 1)
        
        loss_test_vec = []
        loss_vec = []
        for epoch in range(epochs):
            X, Y  = shuffle(X, Y)
            for j in range(n_batchs):
                X_batch = X[j*batch_size:(j+1)*batch_size,...]
                Y_batch = Y[j*batch_size:(j+1)*batch_size,...]
                
                with tf.GradientTape() as tape:
                    y_hat = self.__call__(X_batch)
                    loss = tf.reduce_mean((y_hat - Y_batch)**2)
                loss_vec.append(loss)
                grads = tape.gradient(loss, self.trainable_params)
                self.opt.apply_gradients(zip(grads, self.trainable_params))
                
            
            y_hat_test =  self.__call__(X_test)
            loss_test = tf.reduce_mean((y_hat_test - Y_test)**2)
            loss_test_vec.append(loss_test)
            print("Epoch:", loss)
            
        return loss_vec, loss_test_vec
            
            
        
                
        
