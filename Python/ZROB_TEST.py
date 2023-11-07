from os import path
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from ZROB_ENVS import ZrobEnv
from ZROB_TRAIN import train
import ZROB_PLAY as ZP


def test(env,agent,step_ep=10):

    prev_state = env.reset()
    #print(prev_state)
    episodic_reward = 0
    #time = 0
    step=0
    #step_ep=100
    x1_list=[]
    x2_list=[]
    Zrob=ZP.ZrobPlay()
    while True:
        x1_list.append(prev_state[0])
        x2_list.append(prev_state[1])
        x1=prev_state[0]
        x2=prev_state[1]
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = tf.squeeze(agent(tf_prev_state)).numpy()# type: ignore
        action = np.clip(action, env.action_space.low, env.action_space.high)
        #silence = env.duration-action
        #time += silence
        #print(silence)
        step+=1
        state, reward, done, info = env.step(action)
        Zrob.write_x(x1,x2)
        print(x1,x2)
        f1=1+(x1*4)/12.8
        t1=1/f1
        f2=1+(x2*4)/12.8
        t2=1/f2
        tt=t1+t2
        time.sleep(tt+0.75)
        #print(np.diff(state))
        #print(reward)

        episodic_reward += reward

        if step==step_ep:

            print("test result:{}".format(episodic_reward))
            #print(state)
            plt.figure(1)
            plt.scatter(x1_list,x2_list) # type: ignore
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.show()
            break
        prev_state = state

if __name__ == '__main__':
    repeat = 20
    for i in range(20):
        print("Training model {}".format(i))
        std_dev = np.random.uniform(0.1,0.25)
        critic_lr = np.random.uniform(0.0002/2,0.0002*2)
        actor_lr = np.random.uniform(0.0001/2,0.0001*2)
        tau = np.random.uniform(0.005/2,0.005*2)
        ZROB=train(std_dev = std_dev,critic_lr = critic_lr,actor_lr = actor_lr, tau = tau, actor_name="saved_models/actor{}.keras".format(i))
        model=ZROB.run()
        env=ZROB.env
        test(env,model)