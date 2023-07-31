#include <SoftwareSerial.h>
#include <Servo.h>

SoftwareSerial RaspberrySerial(2, 3); // TX of Raspberry pi, RX of Raspberry pi

Servo ServoThumb;
Servo ServoPointer;
Servo ServoMiddle;
Servo ServoRing;
Servo ServoPinky;
Servo ServoWrist;

void setup() {
  Serial.begin(9600);
  ServoThumb.attach(3);
  ServoPointer.attach(5);
  ServoMiddle.attach(6);
  ServoRing.attach(9);
  ServoPinky.attach(10);
  ServoWrist.attach(11);
  RaspberrySerial.begin(9600);
}
int arr[6];
String temp = "";
int index = 0;
void loop()
{
  if (RaspberrySerial.available())
  {
    String line = RaspberrySerial.readStringUntil('\n');
    //Serial.println(line);
    if (line.startsWith("Servo Values:"))
    {
      index = 0;
      for (int i = 13; i < line.length(); i++)
      {
        if (line[i] != '\r')
        {
          if(line[i]!=',')
          {
            temp+=line[i];       
          }
          else
          {    
            arr[index]=temp.toInt();
            index++;
            temp="";
          }
        }
      }
    }
    /*
    ServoThumb.write(arr[0]);
    ServoPointer.write(arr[1]);
    ServoMiddle.write(arr[2]);
    ServoRing.write(arr[3]);
    ServoPinky.write(arr[4]); 
    ServoWrist.write(arr[5]);
    */
    for(int i = 0; i < index; i++)
    {
      Serial.print("Finger ");
      Serial.print(i);
      Serial.print(": ");
      Serial.println(arr[i]);
    }
    Serial.println("_________________");
  }
}
