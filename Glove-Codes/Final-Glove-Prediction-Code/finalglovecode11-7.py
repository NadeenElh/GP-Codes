import csv
import time
from MCP3008 import MCP3008
import pandas as pd
import smbus
import RPi.GPIO as GPIO

import tensorflow
import numpy as np
from keras.models import load_model
from keras.utils import np_utils

from gtts import gTTS
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

adc = MCP3008()

data1=pd.read_csv(r'nadeen_dataset_8-7-23-normalized.csv')
data2 = pd.read_csv(r'/home/pi/RP_Data/realtime-prediction9-7/lettersEnAr.csv')

G_L=data1['Gesture_label'].drop_duplicates()
G_L=np.array(G_L)

word_button_pin = 23
letter_button_pin = 24

# call model tf lite
#________________________________________________________________________
cnn_model=tensorflow.keras.models.Sequential()   
cnn_model=load_model('CNN_MODLE (4).h5')
cnn_model.load_weights('weights (4).hdf5')


# Setup GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(word_button_pin, GPIO.IN)
#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT = 0x3B
ACCEL_YOUT = 0x3D
ACCEL_ZOUT = 0x3F
GYRO_XOUT  = 0x43
GYRO_YOUT  = 0x45
GYRO_ZOUT  = 0x47

bus = smbus.SMBus(1) # or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68 # MPU6050 device address
word = ""
'''
def speak_word():	
	print(word)
	myjob = gTTS(str(arabic_letter), lang="ar")
	myjob.save("main.mp3")
	play(AudioSegment.from_mp3("main.mp3"))
	word = []
'''
def MPU_Init():
    
    #write to sample rate register
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)

    #Write to power management register
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)

    #Write to Configuration register
    bus.write_byte_data(Device_Address, CONFIG, 0)

    #Write to Gyro configuration register
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)

    #Write to interrupt enable register
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    #Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)
    
        #concatenate higher and lower value
        value = ((high << 8) | low)
        
        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value


MPU_Init()

#GPIO.add_event_detect(word_button_pin, GPIO.FALLING, callback = speak_word, bouncetime=200)

print('Press Ctrl-C to quit...')

while 1:
	GPIO.setup(word_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
	GPIO.setup(letter_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
	x = GPIO.input(word_button_pin)
	y = GPIO.input(letter_button_pin)
	if x == 1:
            MPU_Init()
            c=0
            data_norm=np.array([])
            while 1:
                GPIO.setmode(GPIO.BCM)  # Set's GPIO pins to BCM GPIO numbering
                INDEX_TPIN = 15          # Write GPIO pin number.
                MIDDLE_TPIN = 14
                GPIO.setup(INDEX_TPIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                GPIO.setup(MIDDLE_TPIN, GPIO.IN)
                
                Touch_I = GPIO.input(INDEX_TPIN)
                Touch_M = GPIO.input(MIDDLE_TPIN)
                
                T_S=round(time.time(),4)
                
                c=c+1
                Flex0   = adc.read( channel = 0 )
                Flex1   = adc.read( channel = 1 )
                Flex2   = adc.read( channel = 2 )
                Flex3   = adc.read( channel = 3 )
                Flex4   = adc.read( channel = 4 )
                
                Flex0_norm = (Flex0 - 326)/(670 - 326)
                Flex1_norm = (Flex1 - 559)/(853 - 559)
                Flex2_norm = (Flex2 - 536)/(856 - 536)
                Flex3_norm = (Flex3 - 503)/(835 - 503)
                Flex4_norm = (Flex4 - 500)/(833 - 500)
                
                #Read Accelerometer raw value
                acc_x = read_raw_data(ACCEL_XOUT)
                acc_y = read_raw_data(ACCEL_YOUT)
                acc_z = read_raw_data(ACCEL_ZOUT)

                #Read Gyroscope raw value
                gyro_x = read_raw_data(GYRO_XOUT)
                gyro_y = read_raw_data(GYRO_YOUT)
                gyro_z = read_raw_data(GYRO_ZOUT)

                Ax = acc_x*9.8/16384.0
                Ay = acc_y*9.8/16384.0 
                Az = acc_z*9.8/16384.0

                Gx = gyro_x/131.0
                Gy = gyro_y/131.0
                Gz = gyro_z/131.0

                
                Ax_norm = (Ax - 0.894824)/(19.53062 - 0.894824)
                Ay_norm = (Ay - (-4.98374))/(10.27373 - (-4.98374))
                Az_norm = (Az- (-10.1158))/(8.225684 - (-10.1158))
                
                print(f"F1 = {Flex0} || F2 = {Flex1} || F3 = {Flex2} || F4 = {Flex3} || F5 = {Flex4} || X_accel = {Ax} || Y_accel = {Ay} || Z_accel = {Az} || Touch_I = {Touch_I} || Touch_M = {Touch_M}")
          
                data_norm=np.append(data_norm,[Flex0_norm,Flex1_norm,Flex2_norm,Flex3_norm,Flex4_norm,Ax_norm, Ay_norm, Az_norm,Touch_I, Touch_M],axis=0)
                
                if c==100:
                    data_norm=np.reshape(data_norm,(1,100,10))
                    break
                time.sleep(.01)
                GPIO.cleanup()
            
            

                #prediction 
                #______________________________________________________________

            pred=cnn_model.predict(data_norm).round()
            pred=np.argmax(pred,axis=1)
            #testy_shape=np.argmax(testy_shape,axis=1)
            print(pred)
            print(G_L[pred])

            letter=data2[(data2['index']==int(pred))]
            letter = np.array(letter)
            print(letter)
            arabic_letter = letter[0][1]
            #arabic_letter = arabic_letter[1][1]
            
            print(arabic_letter)
            word+=str(arabic_letter)
            myjob = gTTS(str(arabic_letter), lang="ar")
            myjob.save("main.mp3")
            play(AudioSegment.from_mp3("main.mp3"))
            
            
	elif y == 1:
		print(word)
		myjob = gTTS(str(word), lang="ar")
		myjob.save("main.mp3")
		play(AudioSegment.from_mp3("main.mp3"))
		word = ""
