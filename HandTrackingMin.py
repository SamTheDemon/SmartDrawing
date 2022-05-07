import cv2
import mediapipe as mp
import time

# create the video object
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() # .Hands is defaulty set to 2 hands only
mpDraw = mp.solutions.drawing_utils

#fps
pTime = 0
cTime = 0

# accessing the Camera
while True:
    success, img = cap.read()

    # send our RGB image to hands object
    # first we convert the img to RGB. hands only takes RGb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # we are getting the hands coordinates.
    print(results.multi_hand_landmarks) # shows the hand land marks

    # to check if we have multiple hands and get the info of each hand:

    if results.multi_hand_landmarks:

        for eachHandLandMarks in results.multi_hand_landmarks:
            for id, lm in enumerate(eachHandLandMarks.landmark):
                # print(id, lm)
                height , width, channels = img.shape
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                # print(cx, cy)
                if id == 4: # draw only for id landmark number 4
                    cv2.circle(img, (cx, cy), 15, (255,0,255) , cv2.FILLED)

            # for each single hand
            # mpDraw.draw_landmarks(img, eachHandLandMarks)
            mpDraw.draw_landmarks(img, eachHandLandMarks,
                                  mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)



    cv2.imshow("Image", img)
    if cv2.waitKey(1) & (0XFF == ord('q')):  # q to quit
        break
