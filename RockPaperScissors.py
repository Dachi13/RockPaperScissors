import cv2
import time
import HandTrackingModule as htm
import math


def IdentifyRock(lmList):
    x2, y2 = lmList[4][1], lmList[4][2]
    x1, y1 = lmList[8][1], lmList[8][2]

    x0, y0 = lmList[0][1], lmList[0][2]
    x9, y9 = lmList[9][1], lmList[9][2]

    thumb_pointer_distance = math.hypot(x2 - x1, y2 - y1)
    base_hand_size = math.hypot(x9 - x0, y9 - y0)

    if base_hand_size == 0:
        return

    normalized_distance = thumb_pointer_distance / base_hand_size

    if 0.3 > normalized_distance >= 0.1:  # New normalized thresholds
        print(f'Rock ✊ Detected (normalized: {normalized_distance:.2f})')


def IdentifyScissors(lmList):
    x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
    x2, y2 = lmList[16][1], lmList[16][2]  # Ring Finger
    x3, y3 = lmList[20][1], lmList[20][2]  # Pinky

    thumbRingFingersD = math.hypot(x2 - x1, y2 - y1)
    thumbPinkyD = math.hypot(x1 - x3, y1 - y3)

    allowedDistance = [19, 49]

    if thumbRingFingersD < allowedDistance[0] and thumbPinkyD < allowedDistance[1]:
        print('Scissors')


def IdentifyPaper(lmList):
    thumbDistance = AbsDistanceOfFinger(lmList[4][1], lmList[4][2],
                                        lmList[3][1], lmList[3][2],
                                        lmList[2][1], lmList[2][2])

    indexDistance = AbsDistanceOfFinger(lmList[8][1], lmList[8][2],
                                        lmList[7][1], lmList[7][2],
                                        lmList[6][1], lmList[6][2])

    middleDistance = AbsDistanceOfFinger(lmList[12][1], lmList[12][2],
                                         lmList[11][1], lmList[11][2],
                                         lmList[10][1], lmList[10][2])

    ringDistance = AbsDistanceOfFinger(lmList[16][1], lmList[16][2],
                                       lmList[15][1], lmList[15][2],
                                       lmList[14][1], lmList[14][2])

    pinkyDistance = AbsDistanceOfFinger(lmList[20][1], lmList[20][2],
                                        lmList[19][1], lmList[19][2],
                                        lmList[18][1], lmList[18][2])

    allowedDifference = 0.2

    if (thumbDistance < allowedDifference and indexDistance < allowedDifference and middleDistance < allowedDifference
            and ringDistance < allowedDifference and pinkyDistance < allowedDifference):
        print('paper')


def AbsDistanceOfFinger(x1, y1, x2, y2, x3, y3):
    thumbToBottom = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    thumbToMiddle = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    middleToBottom = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    distanceBySeperationOfThumb = thumbToMiddle + middleToBottom
    # print(f'thumb to middle \t{thumbToMiddle} \tthumb to bottom \t{thumbToBottom}\t thumb to bottom {thumbToBottom} \t thumb to bottom with adding {thumbToMiddle + middleToBottom}')
    return abs(distanceBySeperationOfThumb - thumbToBottom)


wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detectionCon=0.7, maxHands=2)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        IdentifyRock(lmList)
        IdentifyScissors(lmList)
        IdentifyPaper(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
