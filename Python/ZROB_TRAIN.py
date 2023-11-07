from os import path
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

from ZROB_ENVS import ZrobEnv
import ZROB_AGENT as Z
import ZROB_EAR as ZE
import ZROB_PLAY as ZP
import threading
import time
import librosa


class train:
    def __init__(self,std_dev = 0.05,critic_lr = 0.002,actor_lr = 0.001, tau = 0.005, actor_name="saved_models/ZROB5.keras"):
        self.env = ZrobEnv(DoF=4)
        self.V1=[]
        self.V2=[]
        
        self.std_dev = std_dev
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.actor_name = actor_name
        self.tau = tau
        self.ear = ZE.ZrobEar(wind=1.15)
        th1 = threading.Thread(target=self.ear.listen)
        th1.start()
        time.sleep(1)
       
        self.ist = False
        self.Zrob = ZP.ZrobPlay()
        time.sleep(3)
        print("READY TO LEARN")
        #plt.figure(1)
        #y=np.zeros((int(self.ear.args.window * self.ear.args.samplerate / (1000 * self.ear.args.downsample)),))
        #print(y.shape)
        #y=self.ear.get_obs()[:,0]
        #sr = self.ear.args.samplerate
        #plt.plot(np.arange(len(self.ear.get_obs()))/self.ear.args.samplerate,y)
        #plt.show()
        #print(y.shape)
        #onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        #print(onset_envelope.shape)
        #y=self.ear.get_obs()
        #plt.figure(2)
        #plt.plot(onset_envelope)
        #plt.show()


    def readmidi(self):
        V1 = []
    
        V1.append([1,250,0])
        V1.append([2,250,0])
        V1.append([1,250,0])
        V1.append([2,250,0])
        self.V1=V1


    def rew(self):
        y = self.ear.get_obs()[:,0]
        sr = self.ear.args.samplerate
        hop_length = 512
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        #for i in range(len(onset_envelope)):
        #    if onset_envelope[i]<-8:
        #        onset_envelope[i]=0
        onset_frames = librosa.util.peak_pick(onset_envelope, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.3, wait=15)
        #print(onset_frames)
        t = np.linspace(0, len(y)/float(sr), len(onset_envelope))
        if len(onset_frames)==4:
            #er1=((t[onset_frames[1]]-t[onset_frames[0]])-(t[onset_frames[2]]-t[onset_frames[1]]))**2
            #er2=((t[onset_frames[4]]-t[onset_frames[3]])-(t[onset_frames[5]]-t[onset_frames[4]]))**2
            er1=((t[onset_frames[1]]-t[onset_frames[0]])-0.25)**2
            er2=((t[onset_frames[2]]-t[onset_frames[1]])-0.25)**2
            er3=((t[onset_frames[3]]-t[onset_frames[2]])-0.25)**2
            #er4=0.1*(onset_envelope[onset_frames[0]]-onset_envelope[onset_frames[1]])**2
            reward=-500*(er1+er2+er3)
            #print(t[onset_frames[-1]]-t[onset_frames[-2]])
            #er=(2*(t[onset_frames[-1]]-t[onset_frames[-2]])-1)**2
            #print(t[onset_frames[-1]],t[onset_frames[-2]])
            #reward=-100*er
            #print("reward={}".format(reward))
        else:
            reward=-10
        #print(t[onset_frames[-1]]-t[onset_frames[-2]])
        #plt.figure()
        #plt.plot(t, onset_envelope)
        #plt.vlines(t[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.7)
        #plt.show()
        return reward


    def run(self):
        env = self.env
        #th2=threading.Thread(target=self.zarb)
        #th2.start()
        self.readmidi()
        
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
        step_ep = 10
        buffer = Z.Buffer(100000, 64, num_states, num_actions)
        ep_reward_list = []
        avg_reward_list = []
        total_episodes = 10
        total_step = 2000
        gamma = 0.99
        #th2 = threading.Thread(target=self.zarb)
        #th2.start()

        for ep in range(total_episodes):

            prev_state = env.reset()
            
            episodic_reward = 0
            step = 0
           

            while True:

                self.V2=prev_state*25
                
              

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = Z.policy(tf_prev_state, ou_noise, actor_model, lower_bound, upper_bound)
                #silence = env.duration-action
                #time += silence
                #count += 1
                step += 1
                # Recieve state and reward from environment.
                state, reward, done, info = env.step(action)
                self.Zrob.write_mids(self.V1,self.V2)
                #print(prev_state)
                time.sleep(1.3)
                reward=self.rew()

                            
                #if abs(reward-reward2)<0.1:
                #    reward=reward2
                #print(t1)
                #reward2=-100*((((1/(1+(x1*4)/12.8))+(1/(1+(x1*4)/12.8)))-1)**2)
                #print("instant reward = {}".format(reward))
                print("reward2={}".format(reward))

                buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                buffer.learn(actor_model, critic_model,target_actor, target_critic, actor_optimizer, critic_optimizer, gamma)
                Z.update_target(target_actor.variables, actor_model.variables, tau)
                Z.update_target(target_critic.variables, critic_model.variables, tau)

                # End this episode when `done` is True
                if step>=step_ep:
                    #print(time)
                    #print(np.diff(state))
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-2:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)
            #if step>total_step:
                #break
        
        plt.plot(avg_reward_list)
        self.ear.deaf=True
        self.ist=True
        plt.show()
        
        
        target_actor.save(self.actor_name)
        return target_actor


if __name__ == '__main__':
    ZROB = train()
    model=ZROB.run()

