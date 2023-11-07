import serial

class ZrobPlay():
    def __init__(self):
        self.arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

    def write_mids(self, VV1, VV2, VV3, VV4, VV5):
    
        code=201
        code201 = int(code).to_bytes(1, byteorder='big',signed=False)

        len1 = int(len(VV1)).to_bytes(1, byteorder='big',signed=False)
        self.arduino.write(len1)
        for i in range(len(VV1)):        
            for j in range(4):
                V = int(VV1[i][j]).to_bytes(1, byteorder='big',signed=False)
                self.arduino.write(V)
        
        len1 = int(len(VV2)).to_bytes(1, byteorder='big',signed=False)
        self.arduino.write(len1)
        for i in range(len(VV2)):        
            for j in range(4):
                V = int(VV2[i][j]).to_bytes(1, byteorder='big',signed=False)
                self.arduino.write(V)
           
        len1 = int(len(VV3)).to_bytes(1, byteorder='big',signed=False)
        self.arduino.write(len1)
        for i in range(len(VV3)):        
            for j in range(10):
                V = int(VV3[i][j]).to_bytes(1, byteorder='big',signed=False)
                self.arduino.write(V)
      
        len1 = int(len(VV4)).to_bytes(1, byteorder='big',signed=False)
        self.arduino.write(len1)
        for i in range(len(VV4)):        
            for j in range(4):
                V = int(VV4[i][j]).to_bytes(1, byteorder='big',signed=False)
                self.arduino.write(V)
      
        len1 = int(len(VV5)).to_bytes(1, byteorder='big',signed=False)
        self.arduino.write(len1)
        for i in range(len(VV5)):        
            for j in range(4):
                V = int(VV5[i][j]).to_bytes(1, byteorder='big',signed=False)
                self.arduino.write(V)
       
        self.arduino.write(code201)

    def write_x(self,x1,x2):
        
        code=203
        code203 = int(code).to_bytes(1, byteorder='big',signed=False)

        X1 = int(x1*20).to_bytes(1, byteorder='big',signed=False)
        X2 = int(x2*20).to_bytes(1, byteorder='big',signed=False)
        self.arduino.write(X1)
        self.arduino.write(X2)
        self.arduino.write(code203)

