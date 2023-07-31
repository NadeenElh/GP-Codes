#!/usr/bin/env python
# -*- coding: utf-8 -*-

import serial
import time    
import pandas as pd 
import numpy as np
import speech_recognition as sr
import RPi.GPIO as GPIO


# Setup GPIO pins
GPIO.setmode(GPIO.BCM)

# function main 

ser = serial.Serial ("/dev/ttyAMA0", 9600)    #Open port with baud rate

data = pd.read_csv(r'/home/pi/servo_code/new-servo-4-7-2023/servo_dataset4-7-2023.csv')
data_words = pd.read_csv(r'/home/pi/servo_code/new-servo-4-7-2023/servo-word-dataset15-7.csv')

data = data[['Servo_Thumb', 'Servo_Pointer', 'Servo_Middle', 'Servo_Ring', 'Servo_Pinky', 'Servo_Wrist']]
data = np.array(data)
data = data.reshape(30, 6)

#Button GPIO Set Up
Letter_BTN = 14
Word_BTN = 15
GPIO.setup(Letter_BTN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(Word_BTN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
Letter_Signal = GPIO.input(Letter_BTN)
Word_Signal = GPIO.input(Word_BTN)

while 1:
    if(Letter_Signal == 1):
        r = sr.Recognizer()
        with sr.Microphone() as src:
            print('Say something....')
            audio = r.listen(src)
        try:
            t = r.recognize_google(audio,language='ar-AR')
            print(t)                # print word
            for index in range (len(t)):
                print(t[index])
                x=','.join([str(i) for i in data[Letter]])
                ser.write("Servo Values:".encode("utf-8"))
                ser.write(x.encode("utf-8"))
                ser.write(",\r\n".encode("utf-8"))    
    elif(Word_Signal == 1):
        r = sr.Recognizer()
        with sr.Microphone() as src:
            print('Say something....')
            audio = r.listen(src)
        try:
            t = r.recognize_google(audio,language='ar-AR')
            print(t)
            for index in range (len(t)):
                print(t[index])
                x=','.join([str(i) for i in data2[Letter]])
                ser.write("Servo Values:".encode("utf-8"))
                ser.write(x.encode("utf-8"))
                ser.write(",\r\n".encode("utf-8"))    


    """        
    except sr.UnknownValueError as U:
        print(U)
    except sr.RequestError as R:
        print(R)
    """
    #Letter = int(input('Enter the index of the letter: '))
    #servo_values = str(data[Letter])

    
