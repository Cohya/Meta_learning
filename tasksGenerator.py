
import tensorflow as tf 
import numpy as np 
from utils import all_combination, one_hot_encode, create_one_hot

class TasksGeneratorCfar10(object):
    
    def __init__(self):
        
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = tf.keras.datasets.cifar10.load_data()
        
        self.Y_test = self.Y_test[:,0]
        self.Y_train = self.Y_train[:,0]
        
        self.X_train, self.X_test = self.X_train/ 255.0 , self.X_test / 255.0
        
        ## One hot encoding 
        classes = max(len(np.unique(self.Y_test)), len(np.unique(self.Y_train)))
        
        self.Y_train_encoded = one_hot_encode(self.Y_train, classes)
        self.Y_test_encoded  = one_hot_encode(self.Y_test, classes)
        
        ### creating all tasks operation combination 
        # I would like to classify between 4 classes
        # I know that I have only 10 classes in general [0,1,2,3,...9]
        # I will svae classes [6,7,8,9] as my varification over unseen data 
        self.K = 4 
        self.all_combinatoin = all_combination([0,1,2,3,4,5], self.K)  
        self.n_possible_tasks = len(self.all_combinatoin)
    
    def load_specific_task(self, task_num):
        classes_for_task = self.all_combinatoin[task_num]
        
        
        for i in range(len(classes_for_task)):
            c = classes_for_task[i]
            # print(Y_test_original.shape )
                
            # print(c)
            X_test_c = self.X_test[self.Y_test == c]
            X_train_c = self.X_train[self.Y_train == c]
            
            Y_test_c = create_one_hot(i, len(X_test_c),self.K)
            Y_train_c = create_one_hot(i, len(X_train_c), self.K)
            
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
    
    
    def sample_batch_of_tasks(self, batch_size):
        
        ## generate radom tasks:
            
        indexes = np.random.choice(self.n_possible_tasks, size = batch_size, replace= False)
        
        tasks  = [self.load_specific_task(i) for i in indexes]
        
        info_tasks = [self.all_combinatoin[i] for i in indexes]
        
        return tasks, info_tasks ,indexes
    
# generator = TasksGeneratorCfar10()

# tasks, info = generator.sample_batch_of_tasks(batch_size = 4 )
