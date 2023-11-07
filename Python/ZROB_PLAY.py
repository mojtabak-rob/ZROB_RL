import serial
import time

class ZrobPlay():
    def __init__(self):
        self.arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

    def write_mids(self, VV1, VV2):
    
        code=201
        code201 = int(code).to_bytes(1, byteorder='big',signed=False)

        note_num = int(len(VV1)).to_bytes(1, byteorder='big',signed=False)
        self.arduino.write(note_num)
        #print(note_num)
        for i in range(len(VV1)):        
            for j in range(3):
                V = int(VV1[i][j]).to_bytes(1, byteorder='big',signed=False)
                self.arduino.write(V)
                #print(V)
        
        for i in range(len(VV2)):        
            V = int(VV2[i]).to_bytes(1, byteorder='big',signed=False)
            self.arduino.write(V)
            #print(V)
           
       
        self.arduino.write(code201)
        #print(code201)


if __name__ == '__main__':
    V1 = []
    V2 = []
    
    V1.append([1,250,0])
    V1.append([2,250,0])
    V1.append([1,250,0])
    V1.append([2,250,0])

    V2 = [0,0,0,0]
    ZP = ZrobPlay()
    time.sleep(3)
    ZP.write_mids(V1,V2)
    time.sleep(2)
    
    
