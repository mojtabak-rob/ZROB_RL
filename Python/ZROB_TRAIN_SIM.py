from os import path
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import pretty_midi

from ZROB_ENVS import ZrobEnv2
import ZROB_AGENT as Z
#import ZROB_EAR as ZE
#import ZROB_PLAY as ZP
import threading
import time
import librosa


class train:
    def __init__(self,std_dev = 0.05,critic_lr = 0.002,actor_lr = 0.001, tau = 0.005, actor_name="saved_models/SIM1_5D.keras"):
        self.env = ZrobEnv2(DoF=5)
        
        self.std_dev = std_dev
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.actor_name = actor_name
        self.tau = tau
    


    def run(self):
        env = self.env
        #th2=threading.Thread(target=self.zarb)
        #th2.start()
        num_states = env.observation_space.shape[0]# type: ignore
        #print("Size of State Space ->  {}".format(num_states))
        num_actions = env.action_space.shape[0]# type: ignore
        #print("Size of Action Space ->  {}".format(num_actions))
        upper_state = env.observation_space.low# type: ignore
        lower_state = env.observation_space.high# type: ignore

        upper_bound = env.action_space.high # type: ignore
        lower_bound = env.action_space.low # type: ignore

        #print("Max Value of Action ->  {}".format(upper_bound))
        #print("Min Value of Action ->  {}".format(lower_bound))

        std_dev = self.std_dev
        ou_noise = Z.OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))

        actor_model = Z.get_actor(num_states, num_actions, upper_bound)
        critic_model = Z.get_critic(num_states, num_actions)

        target_actor = Z.get_actor(num_states, num_actions, upper_bound)
        target_critic = Z.get_critic(num_states, num_actions)

        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

        critic_lr = self.critic_lr
        actor_lr = self.actor_lr
        tau = self.tau
        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        step_ep = 100
        buffer = Z.Buffer(100000, 64, num_states, num_actions)
        ep_reward_list = []
        avg_reward_list = []
        total_episodes = 2000
        total_step = 2000000
        gamma = 0.99
        #th2 = threading.Thread(target=self.zarb)
        #th2.start()

        for ep in range(total_episodes):

            prev_state = env.reset()
            #print(prev_state)
            episodic_reward = 0
            step = 0
            #count = 0

            while True:

                
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = Z.policy(tf_prev_state, ou_noise, actor_model, lower_bound, upper_bound)
         
                step += 1
                # Recieve state and reward from environment.
                state, reward, done, info = env.step(action)
       
                buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                buffer.learn(actor_model, critic_model,target_actor, target_critic, actor_optimizer, critic_optimizer, gamma)
                Z.update_target(target_actor.variables, actor_model.variables, tau)
                Z.update_target(target_critic.variables, critic_model.variables, tau)

                # End this episode when `done` is True
                if step==step_ep:
                    #print(time)
                    #print(np.diff(state))
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)
            #if step>total_step:
                #break
        
        plt.plot(avg_reward_list)
        plt.show()
   
        target_actor.save(self.actor_name)
        return target_actor


if __name__ == '__main__':
    ZROB = train()
    model=ZROB.run()

