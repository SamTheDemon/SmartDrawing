import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

############################
brushThickness = 15
eraserThickness = 50
###########

# min  from YouTube
# this draws only for the right hand not the left one.
folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overLayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)
# print(len(overLayList))

header = overLayList[0]
drawColor = (0, 0, 0)  # default

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.9)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1- import the image
    success, img = cap.read()
    # flipping the image because when drawing on the left it will draw on the right
    img = cv2.flip(img, 1)
    # 2- find the landmarks from the built hand tracking module.
    img = detector.findHands(img)
    lmList = detector.findPostion(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        # middle finger
        x2, y2 = lmList[12][1:]

        # 3- check which fingers are up from the finger Counting module
        fingers = detector.fingersUp()
        # print(fingers) #show the list which finger is up

        # 4- selections mode.
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            # checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overLayList[0]
                    drawColor = (0, 250, 0)  # black
                elif 550 < x1 < 750:
                    header = overLayList[1]
                    drawColor = (0, 0, 255)  # Red
                elif 800 < x1 < 950:
                    header = overLayList[2]
                    drawColor = (255, 0, 0)  # BLUE
                elif 1050 < x1 < 1200:
                    header = overLayList[3]
                    header = overLayList[2]
                    drawColor = (255, 255, 255)  # white
                cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED)

        # 5- selections mode.
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
        xp, yp = x1, y1

    #     - erase - selection- and drawing

    # layering our images
    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting header image
    img[0:100, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow('Image', img)
    #cv2.imshow('drawing', imgCanvas)
    #cv2.imshow('drawing', imgInv)
    cv2.waitKey(1)
