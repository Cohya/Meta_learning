
import numpy as np 
import tensorflow as tf 
from sklearn.utils import shuffle
from tqdm import tqdm
import time 

class MAML():
    
    def __init__(self, net, tasksGenerator, copy_net = None):
                 # k = None):
         
        #net ---> is a nueral network with accessiblility to the trainable weights ! 
        # trainable_weights = net.trianable_weights 
        self.net = net
        
        if copy_net is not None:
            self.copy_net = copy_net
        # self.tasks = tasks
        self.tasksGenerator = tasksGenerator
        self.trainable_params = self.net.trainable_params
        

        self.number_of_trainable_params = len(self.trainable_params)
        
        self.num_meta_rounds = 300
        self.batch_size = 32
        
        self.meta_opt = tf.keras.optimizers.Adam(learning_rate= 0.001)
        
    def train(self, iterations,
              optimizers,
              loss_fucntion,
              samples_per_class = 10,
              batch_size = 8,
              kappa = 0.01,
              eta = 0.01):
        
        ### In the original paper:
            # Iteration is the number of time we sample random tasks batchs ("epoch style")
            # eta == alpha 
            # kappa == beta 
            # loss functions --> a list of loss function for each task
            # optimizers --> a list of loss fucntion for each task, for now it is not possible 
        self.alpha = eta
        print("Start training in:")
        for i in range(6):
            print(5-i, end="\r")
            time.sleep(1)
        
        self.kappa = kappa
        self.eta = eta
        self.optimizers = optimizers
        loss_func = loss_fucntion
        # self.loss_fucntion = loss_fucntions
        batch_size = batch_size
        self.k = batch_size
        if self.tasksGenerator is not None:
            try:
                self.tasksGenerator.n_possible_tasks
            except:
                print("No modul tasksGenerator.num_of_tasks!")    
        else: 
            K = len(self.tasks)
        samples_per_class = samples_per_class
        for ik in tqdm(range(iterations)):
            all_theta_second_grads = []
            if self.tasksGenerator is not None:
                self.tasks,_,indexes = self.tasksGenerator.sample_batch_of_tasks(batch_size,
                                                                                 samples_per_class)
                
            for i in range(batch_size):
                task = self.tasks[i]
                 #self.loss_fucntions[indexes[i]]
                opt = None # self.optimizers[indexes[i]]
                
                X_train, Y_train, X_test, Y_test = task
                # we should take an equivalent number from each class
            
                ## Shuffle before training 
                X_train, Y_train = shuffle(X_train, Y_train)
                X_test, Y_test = shuffle(X_test, Y_test)

                gradients_2 =  self.calc_grads(X = X_train,
                                               Y = Y_train,
                                               X_test = X_test, 
                                               Y_test = Y_test, 
                                               optimizer = opt,
                                               loss_func = loss_func)
                
                all_theta_second_grads.append(gradients_2) # Now I have all the phi's
        
            self.update_theta(all_theta_second_grads = all_theta_second_grads,
                              indexes = indexes)
            
        #     print(gradients_2)
        #     print(self.trainable_params[-1])
        #     input("dfsdfsdf")
        # print("Iteration:", ik, "Examined Tasks:", indexes)
        return [w.numpy() for w in self.trainable_params]
        
    
    def loss_task(self, X, Y, loss_func, use_copy_net = False):
        ## break it into several runs and averaged it  
        batch_size = 32
        n = len(X)
   
        n_baches = n // batch_size
        if n_baches == 0:
            y_hat = self.net(X, train = True)
            loss = loss_func(y_hat, Y)
        else:
            
            for i in range(n_baches):
                X_batch = X[i*batch_size : (i+1)*batch_size]
                Y_batch  = Y[i*batch_size:(i+1) * batch_size]
                
                if use_copy_net == True:
                    y_hat = self.copy_net(X_batch, train = True)
                else:
                    y_hat = self.net(X_batch, train = True)
                    
                if i == 0 :
                    loss = loss_func(Y_batch, y_hat) * batch_size
                    
                else:
                    # loss += loss_func(Y_batch, y_hat) * batch_size
                    loss = 1/(i+1) * (loss_func(Y_batch, y_hat) * batch_size - loss)
                    # print(loss," ", i, "/", n_baches)
            
            # loss = loss / n 

        return loss
    
    
    def calc_grads(self,  X, Y, X_test, Y_test, optimizer, loss_func)   :
        
        ## Save the original theta 
        self.save_original_thetas()
        
        with tf.GradientTape() as tape2:
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                
                loss_k = self.loss_task(X = X,
                                        Y = Y,
                                        loss_func = loss_func)
                
            gradients = tape.gradient(loss_k, self.trainable_params)
            
            counter = 0
            for counter in range(self.number_of_trainable_params):
                self.copy_net.trainable_params[counter] = self.net.trainable_params[counter] - self.alpha * gradients[counter]
                
            
            self.copy_net.copy_weights()
            
            loss_k2 = self.loss_task(X= X_test, Y = Y_test, loss_func=loss_func, use_copy_net=True)
                
        gradients_2 = tape2.gradient(loss_k2, self.trainable_params)
             
        
        return gradients_2
    
        
    def update_theta(self, all_theta_second_grads, indexes):
        
         #all_theta_second_grads <--- this is a vector of all second derivatives of theta 
        for i in range(self.k):
            if i == 0 :
                seconds_derivatives = all_theta_second_grads[i]
                
            else:
                for j in range(self.number_of_trainable_params):
                    seconds_derivatives[j] += all_theta_second_grads[i][j]
        
            
            
        seconds_derivatives = [ww/self.k  for ww in seconds_derivatives]
        self.meta_opt.apply_gradients(zip(seconds_derivatives, self.trainable_params))
    
        
    def save_original_thetas(self):
        self.original_theta = [w.numpy() for w in self.trainable_params]
        
    def copy_weights_to_net(self, weights_to_copy):
        for w, w_2 in zip(self.trainable_params,weights_to_copy):
            w.assign(w_2)        
        