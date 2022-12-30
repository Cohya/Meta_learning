
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




from Scheduling_Environment import Scheduling_Environment
import numpy as np 
from DDPG.DDPG import DDPG
from DDPG.Nets import ActorNet, CriticNet
from DDPG.ReplayMemory import ReplayMemory
import argparse
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib
from utils import smooting, test_agent, record_agent, main_statistic
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--num_train_episodes', type=int, default=200)
parser.add_argument('--decay', type = float, default = 0.995)

args = parser.parse_args()


env = Scheduling_Environment()

## DEFINE NETS ACTOR - CRITIC 
observation_dims = env.observation_space_n

mu_Net  = ActorNet(observation_dims = env.observation_space_n , action_dims = 1)
mu_Net_target  = ActorNet(observation_dims = env.observation_space_n, action_dims = 1)

q_Net =  CriticNet(observation_dims= observation_dims, action_dims=1)
q_Net_target =  CriticNet(observation_dims= observation_dims, action_dims=1)

batch_size = 64
rm = ReplayMemory(capacity = int(50000) , number_of_channels = env.observation_space_n,
                 agent_history_length = 1, batch_size = batch_size) # 32


ddpg = DDPG(mu_Net = mu_Net, mu_Net_targ= mu_Net_target, 
            q_Net=q_Net, q_Net_targ= q_Net_target,  replay_memory= rm,
           action_clip=[1.5,50], 
           gamma = args.gamma,
           decay = args.decay)

num_train_episodes = 500 
start_steps = 200 # After that number of steps start the trainig (till then sample random action from action space )
num_steps = 0

q_loss_vec = []
mu_loss_vec = []
accumulat_r_vec = []
average_vec_delay1 = []
average_vec_delay2 = []
average_vec_delay3 = []

s = env.reset()
done = False

while not done:
    print("State:", s)
    action = input('Insert action:')
    s_tag, r, done, info = env.step(action)
    s = s_tag
    
    print("Reward:", r, )












