#include <mcp_can.h>
#include <SPI.h>


// Board setup (Arduino Uno with Sparkfun CANBUS Shield)
#define SERIAL Serial

const int SPI_CS_PIN = 10;

MCP_CAN CAN(SPI_CS_PIN);


// ZRob address definition
long ZROB1 = 0x141;
long ZROB2 = 0x145;
long ZROB3 = 0x142;
long ZROB4 = 0x143;

// Initiall and resting position of the robots
const long ZROBIDLE1 = 36000;
const long ZROBIDLE2 = 19000;
const long ZROBIDLE3 = 1000;
const long ZROBIDLE4 = 1000;

// Global variable for position control
long ZROBPos1 = ZROBIDLE1;
long ZROBPos2 = ZROBIDLE2;
long ZROBPos3 = ZROBIDLE3;
long ZROBPos4 = ZROBIDLE4;

// Global bufer for CAN packets
unsigned char len = 0;
unsigned char buf[8];


// Moves a ZRob to a specified position in the defined position
void move (long bot, long pos, bool slow = true) {
  buf[0]= 0xA4;
  buf[1]= 0x00;
  buf[2]= slow ? 0x64: 0x00;
  buf[3]= slow ? 0x00: 0x50;
  buf[4]= pos;
  buf[5]= pos >> 8;
  buf[6]= pos >> 16;
  buf[7]= pos >> 24;
  CAN.sendMsgBuf(bot,0,8,buf);

  while (true) {
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      //CAN.readMsgBuf(&len,buf);
      break;
    }
  }
}

// Moves a ZRob with velocity command
void spd (long bot, long vel) {
  buf[0]= 0xA2;
  buf[1]= 0x00;
  buf[2]= 0x00;
  buf[3]= 0x00;
  buf[4]= vel;
  buf[5]= vel >> 8;
  buf[6]= vel >> 16;
  buf[7]= vel >> 24;
  CAN.sendMsgBuf(bot,0,8,buf);

  while (true) {
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      //CAN.readMsgBuf(&len,buf);
      break;
    }
  }
}

// Modified move function for initial position
void move_init (long bot, long pos, bool slow = true) {
  buf[0]= 0xA4;
  buf[1]= 0x00;
  buf[2]= slow ? 0x64: 0x58;
  buf[3]= slow ? 0x00: 0x02;
  buf[4]= pos;
  buf[5]= pos >> 8;
  buf[6]= pos >> 16;
  buf[7]= pos >> 24;
  CAN.sendMsgBuf(bot,0,8,buf);

  while (true) {
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      //CAN.readMsgBuf(&len,buf);
      break;
    }
  }
}

// Move in a sinusoidal trajectory with a given start and start point
void smove(long bott, long A1, long A2){
  bool cyc = true;
  long Pos = 0;
  float frq = 5;
  float freq = 2*frq/1000; 
  Pos = A1;

  unsigned long starttime = millis();
  unsigned long now = 0;

  while (cyc){
    now = millis();
    Pos = long(A1/2 + A2/2 + (A1 - A2)*cos(freq*PI*(now-starttime))/2);
    move(bott, Pos, false);

    if (freq*(now-starttime)>=1){
      cyc=false;
    }
  }
}

// Move ZRob1 and ZRob2 together in a sinusoidal trajectory
void ssmove2(long A1, long A2, long B, long BB)
{
  bool cyc = true;
  long Pos1 = 0;
  long Pos2 = 0;
  float frq = 9;
  float freq = 2*frq/1000; 
  Pos1 = A1;
  Pos2 = B;

  unsigned long starttime = millis();
  unsigned long now = 0;

  while (cyc){
    now = millis();
    Pos1 = long(A1/2 + A2/2 + (A1 - A2)*cos(freq*PI*(now-starttime))/2);
    move(ZROB1, Pos1, false);

    now = millis();
    Pos2 = long(B/2 + BB/2 + (B - BB)*cos(freq*PI*(now-starttime))/2);
    move(ZROB2, Pos2, false);

    if (freq*(now-starttime)>=1){
      cyc=false;
    }

  }
}

// Kick based on position (ZRob1 & ZRob2)
void kick (long bott, float kick_freq, long Amp)
{
  bool cyc = false;
  float freq = 2*PI*kick_freq/1000;

  unsigned long starttime = millis();
  unsigned long now = 0;
  long Pos = 0;
  long idle = 0;
  if (bott==ZROB1){
    Pos = ZROBIDLE1;
    idle = ZROBIDLE1;
    cyc = true;
  }
  if (bott==ZROB2){
    Pos = ZROBIDLE2;
    idle = ZROBIDLE2;
    cyc = true;
  }
  if (Amp>Pos){
    cyc = false;
  }
  starttime = millis();
  while (cyc) {
    now = millis();
    Pos = long(idle - Amp/2 + Amp*cos(freq*(now - starttime))/2);

    move(bott, Pos, false);
    
    if((freq*(now - starttime))>= 2*PI){
      cyc=false;
    }
  }
  
}

// Kick based on velocity (ZRob3 & ZRob4)
void kicks (long bott, float kick_freq, long Amp)
{
  bool cyc = false;
  float freq = PI*kick_freq/500;

  unsigned long starttime = millis();
  unsigned long now = 0;
  
  long vel = 0;
  if (bott==ZROB3){
    
    cyc = true;
  }
  if (bott==ZROB4){
    
    cyc = true;
  }
  if (Amp>4500){
    cyc=false;
  }
  
  starttime = millis();
  while (cyc) {
    now = millis();    
    vel = long( 500*freq*Amp*sin(freq*(now - starttime)));

    spd(bott, vel);
    
    if((freq*(now - starttime))>= 2*PI){
      spd(bott, 0);    
      cyc=false;
    }
    
  }
  spd(bott, 0);
  spd(bott, 0);
  spd(bott, 0);
  spd(bott, 0);
}

// Drum roll with ZRob1 & ZRob2
// Input variables:(roll_amp='Amplitude of motion',...
                    //idle='pick position in roll',...
                    //roll_freq='frequency of motion',...
                    //shift='phase shift between two arms',...
                    //roll_length='drum roll duration')
void roll (long roll_amp1, long roll_amp2, long idle1,
           long idle2, float roll_freq1, float roll_freq2,
            long shift, unsigned long roll_length)
{
  bool cyc = true;
  if ((roll_amp1 + idle1 > 15000)||(roll_amp2 + idle2 > 15000)){
    cyc = false;
  }
  float freq1 = 2*PI*roll_freq1/1000;
  float freq2 = 2*PI*roll_freq2/1000;
  float pshift = shift*PI/180;
  ZROBPos1 = ZROBIDLE1 - idle1;
  ZROBPos2 = long(ZROBIDLE2 - idle2 - roll_amp2/2 + roll_amp2*cos(pshift)/2);
  ssmove2(ZROBIDLE1, ZROBPos1, ZROBIDLE2, ZROBPos2);
  

  unsigned long starttime = millis();
  unsigned long now = 0;

  while (cyc) {
    now = millis();
    ZROBPos1 = long(ZROBIDLE1 - idle1 - roll_amp1/2 +
                    roll_amp1*cos(freq1*(now - starttime))/2);
    ZROBPos2 = long(ZROBIDLE2 - idle2 - roll_amp2/2 +
                    roll_amp2*cos(freq2*(now - starttime) + pshift)/2);

    move(ZROB1, ZROBPos1, false);
    move(ZROB2, ZROBPos2, false);
    
    
    if((now - starttime)>= roll_length){
      cyc=false;
    }
  }
  ssmove2(ZROBPos1, ZROBIDLE1, ZROBPos2, ZROBIDLE2);
}

// Initiallization
void setup() {
  SERIAL.begin(115200);
  delay(500);
  while (CAN_OK != CAN.begin(CAN_1000KBPS))
  {
      //SERIAL.println("CAN BUS Shield init fail");
      //SERIAL.println("Init CAN BUS Shield again");
      delay(100);
  }

  ZROBPos1 = ZROBIDLE1;
  ZROBPos2 = ZROBIDLE2;
  ZROBPos3 = ZROBIDLE3;
  ZROBPos4 = ZROBIDLE4;

  move_init(ZROB1, ZROBPos1, false);
  delay(100);
  
  //move_init(ZROB2, ZROBPos2, false);
  //delay(100);

  //move_init(ZROB3, ZROBPos3, true);
  //delay(100);

  //move_init(ZROB4, ZROBPos4, true);
  //delay(100);

  //kicks(ZROB3, 5, 2000);
  //delay(100);

  //kicks(ZROB4, 5, 2000);
  //delay(100);


}


// Main Loop
void loop() {

  //read the length of ZRob1 kick
  while (!SERIAL.available()); 
  unsigned long X1 = Serial.read(); 

  while (!SERIAL.available()); 
  unsigned long X2 = Serial.read(); 
  
  // Validating SERIAL message
  while (!SERIAL.available());
  unsigned long CODE203 = Serial.read();

  // Execute the received message
  if (CODE203==203)
  {
    kick(ZROB1, 1+X1*4/256, 13000);
    kick(ZROB1, 1+X2*4/256, 13000);
  
  }
}
