import csv
import time
from MCP3008 import MCP3008
import pandas as pd
import smbus
import RPi.GPIO as GPIO


# Setup GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
# Set the servo motor pin as output pin
GPIO.setup(4,GPIO.OUT)

pwm = GPIO.PWM(4,50)
pwm.start(0)

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


adc = MCP3008()

MPU_Init()


#header1=[' ',' ',' ',' ','Flex Sensor',' ',' ',' ',' ']
header2=['Gesture_id','person_id','Count_signals','TimeStamp','Gesture_label','F0','F1','F2','F3','F4', 'Accel_x', 'Accel_y', 'Accel_z', 'Gyro_x', 'Gyro_y', 'Gyro_z' , 'Index_Touch', 'Middle_Touch', 'Output']
file = open('/home/pi/RP_Data/reading_glove_values/nadeen-dataset-23-6.csv', 'a',newline='')
w = csv.writer(file)

#w.writerow(header1)
w.writerow(header2)

G_ID=int(input("Gesture_ID : "))
P_ID=int(input("Person_ID : "))
G_L=input("Gesture_Label : ")
out=int(input("out : "))
c=0
# header1=[' ',' ',' ',' ','Flex Sensor',' ',' ',' ',' ']
# header2=['Gesture_id','person_id','Count_signals','TimeStamp','Gesture_label','F0','F1','F2','F3','F4']

print('Press Ctrl-C to quit...')
while 1:
    GPIO.setmode(GPIO.BCM)  # Set's GPIO pins to BCM GPIO numbering
    INDEX_TPIN = 15          # Write GPIO pin number.
    MIDDLE_TPIN = 14
    GPIO.setup(INDEX_TPIN, GPIO.IN)
    GPIO.setup(MIDDLE_TPIN, GPIO.IN)
    Touch_I = GPIO.input(INDEX_TPIN)
    Touch_M = GPIO.input(MIDDLE_TPIN)

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


    T_S=round(time.time(),4)
    c=c+1
    Flex0   = adc.read( channel = 0 )
    Flex1   = adc.read( channel = 1 )
    Flex2   = adc.read( channel = 2 )
    Flex3   = adc.read( channel = 3 )
    Flex4   = adc.read( channel = 4 )
    data=[G_ID,P_ID,c,T_S,G_L,Flex0,Flex1,Flex2,Flex3,Flex4, Ax, Ay, Az, Gx, Gy, Gz, Touch_I, Touch_M, out]
    w.writerow(data)
    print(f"F1 = {Flex0} || F2 = {Flex1} || F3 = {Flex2} || F4 = {Flex3} || F5 = {Flex4} || X_accel = {Ax} || Y_accel = {Ay} || Z_accel = {Az} || Touch_I = {Touch_I} || Touch_M = {Touch_M}")
    if c==100:
    #data=pd.read_csv('/home/pi/RP_Data/tst_mohanad_part2/tstp2_1_alef.csv')
        #kk=data[(data['Gesture_id']==G_ID)]
        break
    time.sleep(.01)
    GPIO.cleanup()

file.close()
