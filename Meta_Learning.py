
import numpy as np 
import tensorflow as tf 
from sklearn.utils import shuffle

class FOMAML():
    
    def __init__(self, net, tasks, optimizers, loss_fucntions, k = None):
        # k - is the numebr of tests to train on each run 
        
        self.net = net
        self.tasks = tasks
        self.optimizers = optimizers
        self.loss_fucntions = loss_fucntions
        self.trainable_params = self.net.trainable_params
        self.kappa = 0.01
        self.eta = 0.1
        self.number_of_trainable_params = len(self.trainable_params)
        
        self.num_meta_rounds = 300
        self.batch_size = 64
        
        
        if k is None:
            self.k = len(tasks)//2 
        else:
            self.k = k
            
        
    def train(self, iterations):
        
        for _ in range(iterations):
            phi_s = []
            K = len(self.tasks)
            indexs = np.random.choice(K,size=self.k, replace=False)
            
            for i in indexs:
                task = self.tasks[i]
                loss_func = self.loss_fucntions[i]
                opt = self.optimizers[i]
                
                X_train, Y_train, X_test, Y_test = task
                
                ## Shuffle before training 
                X_train, Y_train = shuffle(X_train, Y_train)
                X_test, Y_test = shuffle(X_test, Y_test)
                print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
                # X_train, Y_train, X_test, Y_test = X_train[:32], Y_train[:32], X_test[:32], Y_test[:32]
                
                
                phi_k = self.calc_phi(X_train = X_train,
                                        Y_train = Y_train,
                                        optimizer = opt,
                                        loss_func = loss_func)
                
                phi_s.append(phi_k) # Now I have all the phi's
        
            self.update_theta(phi_s = phi_s,
                              indexs = indexs)
            
        return [w.numpy() for w in self.trainable_params]
        
        
        
        
    def loss_task(self, X, Y, loss_func):
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
                
                y_hat = self.net(X_batch, train = True)
                if i == 0 :
                    print(y_hat.shape, Y_batch.shape)
                    loss = loss_func(Y_batch, y_hat) * batch_size
                    
                    print(loss)
                else:
                    # loss += loss_func(Y_batch, y_hat) * batch_size
                    loss = 1/(i+1) * (loss_func(Y_batch, y_hat) * batch_size - loss)
                    print(loss," ", i, "/", n_baches)
            
            # loss = loss / n 
            print("pass in loss task")
            input("input")
        return loss
    
    def calc_phi(self, X_train, Y_train, optimizer, loss_func):
        
        gradients_theta = self.calc_grads(X = X_train,
                                    Y = Y_train,
                                    optimizer=optimizer,
                                    loss_func = loss_func)
        
        phi_k = [w - self.eta * g_L_w for w,g_L_w in zip(self.trainable_params, gradients_theta)]
        # print(gradients_theta)
        # input("I just hold you in clacl_phi in FOMAML modul!")
        return phi_k
    
    def calc_grads(self,  X, Y, optimizer, loss_func)   :
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            
            loss_k = self.loss_task(X = X,
                                    Y = Y,
                                    loss_func = loss_func)
            input("Hallo")
        gradients = tape.gradient(loss_k, self.trainable_params)
        
        return gradients
    
    def calc_grad_phi(self, X_test, Y_test, loss_func, optimizer, phi_k):
        # insert the phi into the net 
        
        self.copy_weights_to_net(weights_to_copy = phi_k) # 
        
        g_phi = self.calc_grads(X = X_test, 
                                Y = Y_test,
                                optimizer= optimizer,
                                loss_func= loss_func)
        return g_phi
        
    def save_original_thetas(self):
        self.original_theta = [w.numpy() for w in self.trainable_params]
        
    def copy_weights_to_net(self, weights_to_copy):
        for w, w_2 in zip(self.trainable_params,weights_to_copy):
            w.assign(w_2)
        
    def update_theta(self, phi_s, indexs):
        
        grads_phi_s = []
        self.save_original_thetas()
        for i,j in enumerate(indexs):
            task = self.tasks[j]
            loss_func = self.loss_fucntions[j]
            opt = self.optimizers[j]
            phi_k = phi_s[i]
            _ , _, X_test, Y_test = task 
            
            g_phi_k = self.calc_grad_phi(X_test = X_test,
                                         Y_test = Y_test,
                                         loss_func = loss_func,
                                         optimizer = opt, 
                                         phi_k = phi_k)
            
            
            grads_phi_s.append(g_phi_k)
        # print("##################################################")
        # for i,ww in enumerate(grads_phi_s):
        #     print(i,":", ww)
        # thea_new calculate here the new theta
        sum_of_phis_grads = []
        for i in range(self.number_of_trainable_params):
            w = 0.
            for j in range(self.k):
                w += grads_phi_s[j][i]
        
            sum_of_phis_grads.append(w)
            # print("update shape: ", w.shape)
            # print("i AM HERE:", self.original_theta[i])
            self.original_theta[i] = self.original_theta[i] - self.kappa * w/self.k
        
        self.copy_weights_to_net(weights_to_copy = self.original_theta)
        
        
    
        
        
        