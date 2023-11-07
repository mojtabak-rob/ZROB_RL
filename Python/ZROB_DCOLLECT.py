from os import path
import numpy as np
#import tensorflow as tf
#import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

#from ZROB_ENVS import ZrobEnv2
#import ZROB_AGENT as Z
import ZROB_EAR as ZE
import ZROB_PLAY as ZP
import threading
import time
import librosa

def readmidi(midi_name):
    midi_num = 1
    V1 = []
    V2 = []
    V3 = []
    V4 = []
    V5 = []
    end_time = 0.2
    for i in range(midi_num):        
        PM = []
        PM.append(pretty_midi.PrettyMIDI(midi_name))
        end_time = PM[i].get_end_time()
        for note in PM[i].instruments[0].notes:
            if note.pitch == 50:
                V1.append([10*(note.start),i,150,128])
        
        for note in PM[i].instruments[0].notes:
            if note.pitch == 45:
                V2.append([10*(note.start),i,80,128])
            
        for note in PM[i].instruments[0].notes:
            if note.pitch == 53:
                V3.append([10*(note.start), 10*(note.end), i,
                    50, 100, 130, 100, note.velocity, note.velocity, 180])

        for note in PM[i].instruments[0].notes:
            if note.pitch == 55:
                V4.append([10*(note.start),i,128,32])

        for note in PM[i].instruments[0].notes:
            if note.pitch == 60:
                V5.append([10*(note.start),i,128,32])
    return V1,V2,V3,V4,V5,end_time

class zdata:
    def __init__(self, midi_name='aa1.mid'):
        print("Initializing...")
        self.a=1
        self.V1,self.V2,self.V3,self.V4,self.V5,self.end_time = readmidi(midi_name)
        self.ear = ZE.ZrobEar(wind=self.end_time*2)
        th1 = threading.Thread(target=self.ear.listen)
        th1.start()
        time.sleep(1)
        self.Zrob = ZP.ZrobPlay()
        self.Zrob.write_x(0,0)
        time.sleep(3)
        print("READY TO LEARN")
        self.DATA=[]
        self.inf=[]

    def save(self,i,par):
        y = self.ear.get_obs()[:,0]
        sr = self.ear.args.samplerate
        temp = [i,sr,par]
        self.DATA.append(y)
        self.inf.append(temp)





if __name__ == '__main__':
    ZDATA = zdata(midi_name='dcol.mid')
    for i in range(25):
        for j in range(len(ZDATA.V1)):
            ZDATA.V1[j][3] = 10 + i*10
        for j in range(len(ZDATA.V2)):
            ZDATA.V2[j][3] = 10 + i*10
        ZDATA.Zrob.write_mids(ZDATA.V1,ZDATA.V2,ZDATA.V3,ZDATA.V4,ZDATA.V5)
        time.sleep(ZDATA.end_time)
        ZDATA.save(i=i,par=70 + i*10)

    ZDATA.ear.deaf=True
    #print(ZDATA.DATA)
    #DATA = np.asarray(ZDATA.DATA)
    #inf = np.asarray(ZDATA.inf)
    #print(DATA)
    np.save("dat.npy",ZDATA.DATA)
    np.save("inf.npy",ZDATA.inf)
    test=np.load("dat.npy")
    testinf=np.load("inf.npy")
    y=test[14]
    sr=testinf[14,1]
    hop_length = 512
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    plt.plot(onset_envelope)
    plt.show()
    #print(test[0])    
    
    