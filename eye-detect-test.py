import numpy as np
import time
from scipy import signal
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
centers = []
lastPoint = (0, 0)
counter = 0

def getLeftEye(eyes):
    leftmost = 99999999
    leftEye = [0, 0, 0, 0]
    for eye in eyes:
        if (eye[0] < leftmost):
            leftmost = eye[0]
            leftEye = eye
    return leftEye

def getEyeball(eye, ew, eh, circles):

    stub, listSize, params = circles.shape
    sums = np.zeros(listSize)
    for i in range(listSize):
        cir_x, cir_y, cir_r = circles[0][i]
        # mask_background = np.zeros((ew,eh))
        # mask_background[int(cir_x-cir_r):int(cir_x+cir_r),int(cir_y-cir_r):int(cir_y+cir_r)] = 1
        # out = signal.convolve2d(eye, mask_background)
        # sums[i] = np.sum(out)
        for y in range(eh):
            # row = eye[:,y]
            for x in range(ew):
                # value = row[x]
                if ((np.power(x - cir_x, 2) + np.power(y - cir_y, 2)) < np.power(cir_r, 2)):
                        sums[i] += eye[x][y]
    smallest = np.argmin(sums)
    return circles[0][smallest]

def stabilize(points, windowSize):
    sumX = 0
    sumY = 0
    count = 0
    point_size = np.shape(points)[0]
    start = max(0, point_size - windowSize)
    for i in range(start, point_size):
        sumX += points[i][0]
        sumY += points[i][1]
        count += 1
    if (count > 0):
        sumX /= count
        sumY /= count
    return (int(sumX), int(sumY))

while 1:
    # cap.set(cv2.CAP_PROP_FPS, 2)
    time.sleep(0.1)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draws the rectangle for face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] #color version of face img
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 4, 0)
        if (np.size(eyes) != 0):
            counter = 0
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyeRect = getLeftEye(eyes)
            ex, ey, ew, eh = eyeRect
            eyeCropped = roi_gray[ey:ey+eh, ex:ex+ew]
            # eyeCropped = cv2.equalizeHist(eyeCropped)
            circles = cv2.HoughCircles(eyeCropped, cv2.HOUGH_GRADIENT, 1, int(ew/8), 250, 15, int(eh/8), int(eh/3))
            if (np.size(circles) != 0): # size of circles?
                pupil = getEyeball(eyeCropped, ew, eh, circles)
                centers.append(pupil)
                center = stabilize(centers, 5)
                (cx, cy) = center
                if (np.shape(centers)[0] > 1):
                    (lx, ly) = lastPoint
                    diff_x = (cx - lx)
                    diff_y = (cy - ly) # harder to move in up down
                    # print(diff_x)
                    # print(diff_y)
                    if ((np.absolute(diff_x) > 5) or (np.absolute(diff_y) > 5)):
                        if (np.absolute(diff_x) > np.absolute(diff_y)):
                            # moving left right
                            # video is mirrored so direction changes
                            if (diff_x > 0):
                                print(0)
                            else:
                                print(1)
                        else:
                            # moving up and down
                            if (diff_y > 0):
                                print(3)
                            else:
                                print(2)
                lastPoint = center
                # pupil[2] is radius
                cv2.circle(img, (x + ex + cx, y + ey+ cy), int(pupil[2]), (0,0,255) ,2)
                # cv2.circle(eyeCropped, center, int(pupil[2]), (255,255,255),2)
            #cv2.imshow('Eye', eyeCropped)
        else:
            counter += 1;
            if (counter >= 5):
                print(4)
                counter = 0


    cv2.imshow('img',img)
    if (cv2.waitKey(15) >= 0): 
        break;
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break

cap.release()
cv2.destroyAllWindows()

