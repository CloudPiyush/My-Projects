import cv2
from random import randrange

Trained_Face_Data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   

# Web_Cam = cv2.VideoCapture('C:/Users/Shree/OneDrive/Pictures/Lohgad Tracking/Videos/VID20230806141813.mp4')

# Height = 900
# width  = 700 

# Size = (Height,width)
# Web_Cam = cv2.resize(Web_Cam,Size)

Web_Cam = cv2.VideoCapture(0)

while True:

    Frame_red , Frame = Web_Cam.read()
    Grayscaled_Image = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)

    Detect_Face = Trained_Face_Data.detectMultiScale(Grayscaled_Image)

    for (X,Y,W,H) in Detect_Face:
        cv2.rectangle(Frame,(X,Y),(X+W , Y+H), (randrange(256), randrange(256), randrange(256)),4)

    cv2.imshow('Clever Programmer Face Detector',Frame)
    cv2.waitKey(1)

