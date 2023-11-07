from os import path
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from ZROB_ENVS import ZrobEnv2
from ZROB_TRAIN import train
import ZROB_TEST as ZT



if __name__ == '__main__':
    #ear=ZE.ZrobEar()
    env=ZrobEnv2(DoF=2)
    agent=tf.keras.models.load_model("saved_models/ZROB1.keras")
    prev_state = env.reset()
    episodic_reward = 0
    step=0
    x1_list=[]
    x2_list=[]
    #x3_list=[]
    #x4_list=[]
    #x5_list=[]
    
    
    while True:
        x1_list.append(prev_state[0])
        x2_list.append(prev_state[1])
        #x3_list.append(prev_state[2])
        #x4_list.append(prev_state[3])
        #x5_list.append(prev_state[4])
        
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = tf.squeeze(agent(tf_prev_state)).numpy()# type: ignore
        action = np.clip(action, env.action_space.low, env.action_space.high) # type: ignore
       
        step+=1
        state, reward, done, info = env.step(action)

        episodic_reward += reward

        if step==15:

            print("test result:{}".format(episodic_reward))
            #print(state)
            plt.figure(1)
            plt.plot(x1_list)
            plt.plot(x2_list)
            #plt.plot(x3_list)
            #plt.plot(x4_list)
            #plt.plot(x5_list)
            #plt.scatter([5],[5], marker="x") # type: ignore
            #plt.scatter(x1_list,x2_list) # type: ignore
            #plt.xlim(0,10)
            #plt.ylim(0,10)
            #plt.xlabel("X1")
            #plt.ylabel("X2")
            plt.show()
            break
        prev_state = state