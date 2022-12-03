import datetime
import math
import cv2
import numpy as np

#global variables
width = 0
height = 0
EntranceCounter = 0
ExitCounter = 0
MinCountourArea = 3000  #Adjust ths value according to your usage
BinarizationThreshold = 70  #Adjust ths value according to your usage
OffsetRefLines = 150  #Adjust ths value according to your usage

#Check if an object in entering in monitored zone
def CheckEntranceLineCrossing(y, CoorYEntranceLine, CoorYExitLine):
    AbsDistance = abs(y - CoorYEntranceLine)	

    if ((AbsDistance <= 2) and (y < CoorYExitLine)):
        return 1
    else:
        return 0

#Check if an object in exitting from monitored zone
def CheckExitLineCrossing(y, CoorYEntranceLine, CoorYExitLine):
    AbsDistance = abs(y - CoorYExitLine)	

    if ((AbsDistance <= 2) and (y > CoorYEntranceLine)):
        return 1
    else:
        return 0

camera = cv2.VideoCapture(0)

#force 640x480 webcam resolution
camera.set(3,640)
camera.set(4,480)

ReferenceFrame = None

#The webcam maybe get some time / captured frames to adapt to ambience lighting. For this reason, some frames are grabbed and discarted.
for i in range(0,20):
    (grabbed, Frame) = camera.read()

while True:    
    (grabbed, Frame) = camera.read()
    height = np.size(Frame,0)
    width = np.size(Frame,1)

    #if cannot grab a frame, this program ends here.
    if not grabbed:
        break

    #gray-scale convertion and Gaussian blur filter applying
    GrayFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    GrayFrame = cv2.GaussianBlur(GrayFrame, (21, 21), 0)
    
    if ReferenceFrame is None:
        ReferenceFrame = GrayFrame
        continue

    #Background subtraction and image binarization
    FrameDelta = cv2.absdiff(ReferenceFrame, GrayFrame)
    FrameThresh = cv2.threshold(FrameDelta, BinarizationThreshold, 255, cv2.THRESH_BINARY)[1]
    
    #Dilate image and find all the contours
    FrameThresh = cv2.dilate(FrameThresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(FrameThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    QttyOfContours = 0

    #plot reference lines (entrance and exit lines) 
    CoorYEntranceLine = (height / 2)-OffsetRefLines
    CoorYExitLine = (height / 2)+OffsetRefLines
    cv2.line(Frame, (0,CoorYEntranceLine), (width,CoorYEntranceLine), (255, 0, 0), 2)
    cv2.line(Frame, (0,CoorYExitLine), (width,CoorYExitLine), (0, 0, 255), 2)


    #check all found countours
    for c in cnts:
        #if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < MinCountourArea:
            continue

        QttyOfContours = QttyOfContours+1    

        #draw an rectangle "around" the object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(Frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #find object's centroid
        CoordXCentroid = (x+x+w)/2
        CoordYCentroid = (y+y+h)/2
        ObjectCentroid = (CoordXCentroid,CoordYCentroid)
        cv2.circle(Frame, ObjectCentroid, 1, (0, 0, 0), 5)
        
        if (CheckEntranceLineCrossing(CoordYCentroid,CoorYEntranceLine,CoorYExitLine)):
            EntranceCounter += 1

        if (CheckExitLineCrossing(CoordYCentroid,CoorYEntranceLine,CoorYExitLine)):  
            ExitCounter += 1

    print ("Total countours found: "+str(QttyOfContours))

    #Write entrance and exit counter values on frame and shows it
    cv2.putText(Frame, "Entrances: {}".format(str(EntranceCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.putText(Frame, "Exits: {}".format(str(ExitCounter)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Original Frame", Frame)
    cv2.waitKey(1);


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()