import cv2
import time
import os
import HandTrackingModule as htm
wCam = 640
hCam = 480

cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList=[]
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #import the imgs paths
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

pTime=0


detector =htm.handDetector(detectionCon=0.75)

tipIds = [4, 8 , 12, 16, 20]

while True:
    success, img =cap.read()
    img = detector.findHands(img)

    lmList = detector.findPostion(img,draw=False)
    #print(lmList)

    if len(lmList) !=0:
        fingers = []

        # open-closed thump right hand
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
            # print("index finger is open")
        else:
            fingers.append(0)

        # open - closed other fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
                #print("index finger is open")
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)


       # h, w, c = overlayList[totalFingers-1].shape
       # img[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.putText(img, str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)

    cTime = time.time()
    fps= 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(fps)}', (400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


