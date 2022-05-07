import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon,
                                        self.trackCon)  # .Hands is defaulty set to 2 hands only
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # send our RGB image to hands object
        # first we convert the img to RGB. hands only takes RGb
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # we are getting the hands coordinates.
        # print(results.multi_hand_landmarks)  # shows the hand land marks

        # to check if we have multiple hands and get the info of each hand:
        if self.results.multi_hand_landmarks:
            for eachHandLandMarks in self.results.multi_hand_landmarks:
                # for each single hand
                # mpDraw.draw_landmarks(img, eachHandLandMarks)
                if draw:
                    self.mpDraw.draw_landmarks(img, eachHandLandMarks,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPostion(self, img, handNo=0, draw=True):

        self.landMarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                height, width, channels = img.shape
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                # print(cx, cy)
                self.landMarkList.append([id, cx, cy])
                if draw:
                    # if id == 4:  # draw only for id landmark number 4
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.landMarkList

    def fingersUp(self):
        fingers = []

        # open-closed thump right hand
        if  self.landMarkList[self.tipIds[0]][1] <  self.landMarkList[self.tipIds[0] - 1][1]:
            fingers.append(1)
            # print("index finger is open")
        else:
            fingers.append(0)


        # open - closed other fingers
        for id in range(1, 5):
            if  self.landMarkList[self.tipIds[id]][2] <  self.landMarkList[self.tipIds[id] - 2][2]:
                fingers.append(1)
                # print("index finger is open")
            else:
                fingers.append(0)

        return fingers
def main():
    # create the video object
    cap = cv2.VideoCapture(0)

    # fps
    pTime = 0
    cTime = 0

    detector = handDetector()

    # accessing the Camera
    while True:
        success, img = cap.read()

        # send the img
        img = detector.findHands(img)
        landMarkList = detector.findPostion(img)
        if len(landMarkList) != 0:
            print(landMarkList[4])  # change the number of for 21l landmamkrs

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
