import numpy as np
import cv2
import textwrap
import time
from scipy import signal

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
centers = []
lastPoint = (0, 0)
counter = 0

#### EYE-DETECTION:
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

font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale 
fontScale = 2
color = (0,0,0)
thickness = 4 # font 2px

window_name = 'Writer'
text = 'Input some text: '

KEYBOARD_NORMAL = cv2.imread('img/keyreg.png')
KEYBOARD_SHIFT = cv2.imread('img/keyshift.png')


key_width = 900
cam_height = 220 # ALSO FOR TEXT HEIGHT

h, w, _ = KEYBOARD_NORMAL.shape
scale = key_width * 1.0 / w
print(scale)
KEYBOARD_NORMAL = cv2.resize(KEYBOARD_NORMAL, (int(scale*w), int(scale*h)))

h, w, _ = KEYBOARD_SHIFT.shape
scale = key_width * 1.0 / w
KEYBOARD_SHIFT = cv2.resize(KEYBOARD_SHIFT, (int(scale*w), int(scale*h)))

print(KEYBOARD_NORMAL.shape)
print(KEYBOARD_SHIFT.shape)

textRatio = 2.0 / 3.0

cursorPos = ()

isShift = False

keyboard = None
keydict = None

user_keyrow = 0
user_keycol = 0
r = 0
c = 0
MAXROWS = 4
MAXCOLS = 14

cursor_x = user_keyrow * 59 + 41
cursor_y = user_keycol * 62 + 240
cursor_len = 52

def updateCursor(r, c):
    cursor_x = c * 59 + 41
    cursor_y = r * 62 + 240
    return (cursor_x, cursor_y)

def updateDirection(code, r, c, keydict, text, isShift):
    if code == 0: # LEFT
        c = max(c-1, 0)
    elif code == 1: # RIGHT
        c = min(c+1, MAXCOLS - 1)
    elif code == 2: # UP
        r = max(r-1, 0)
    elif code == 3: # DOWN
        r = min(r+1, MAXROWS - 1) 
    elif code == 4:
        text, isShift = enterCharacter(r, c, keydict, text, isShift)
    else:
        print("Uh oh! Where are you looking?")
    return (r, c, text, isShift)

def enterCharacter(r, c, dic, text, isShift):
    char = dic[r][c]
    if char == 'SHIFT':
        isShift = not isShift
        return text, isShift
    text += char
    print(char)
    return text, isShift

## INITIALIZE REGULAR KEYBOARD
r1 = ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '[', ']', '\n']
r2 = ['TAB', '\'', ',', '.', 'P', 'Y', 'F', 'G', 'C', 'R', 'L', '/', '=', '\\']
r3 = ['CAPS', 'A', 'O', 'E', 'U', 'I', 'D', 'H', 'T', 'N', 'S', '-', '\n', 'CMD']
r4 = ['SHIFT', ';', 'Q', 'J', 'K', 'X', 'B', 'M', 'W', 'V', 'Z', ' ', 'CTRL', 'ALT']

keydict_reg = [r1, r2, r3, r4]

## INITIALIZE SHIFTED KEYBOARD
r1 = ['~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '{', '}', '\n']
r2 = ['TAB', '"', '<', '>', 'P', 'Y', 'F', 'G', 'C', 'R', 'L', '?', '+', '|']
r3 = ['CAPS', 'A', 'O', 'E', 'U', 'I', 'D', 'H', 'T', 'N', 'S', '_', '\n', 'CMD']
r4 = ['SHIFT', ':', 'Q', 'J', 'K', 'X', 'B', 'M', 'W', 'V', 'Z', ' ', 'CTRL', 'ALT']

keydict_shift = [r1, r2, r3, r4]

input_dir = None

while 1:
    if isShift:
        keyboard = KEYBOARD_SHIFT
        keydict = keydict_shift
    else:
        keyboard = KEYBOARD_NORMAL
        keydict = keydict_reg

    # clear rectangle every loop
    rect = cv2.imread('img/empty_rect.png')
    #rect = cv2.resize()

    time.sleep(0.1)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if (np.size(eyes) != 0):
            counter = 0
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyeRect = getLeftEye(eyes)
            ex, ey, ew, eh = eyeRect
            eyeCropped = roi_gray[ey:ey+eh, ex:ex+ew]

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
                                input_dir = 0
                                print(0)
                            else:
                                input_dir = 1
                                print(1)
                        else:
                            # moving up and down
                            if (diff_y > 0):
                                input_dir = 3
                                print(3)
                            else:
                                input_dir = 2
                                print(2)
                lastPoint = center
                # pupil[2] is radius
                cv2.circle(img, (x + ex + cx, y + ey+ cy), int(pupil[2]), (0,0,255) ,2)
                # cv2.circle(eyeCropped, center, int(pupil[2]), (255,255,255),2)
            #cv2.imshow('Eye', eyeCropped)
        else:
            counter += 1;
            if (counter >= 5):
                input_dir = 4
                print(4)
                counter = 0

    # RENDER TEXT
    lines = textwrap.wrap(text, 45)
    x_coord = 20
    y_coord = 50
    for line in lines:
        org = (x_coord, y_coord)
        rect = cv2.putText(rect, line, org, font, fontScale,  
                 color, thickness, cv2.LINE_AA, False)
        y_coord += 70

    # RESIZING
    scale = cam_height * 1.0 / img.shape[0]
    dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, dim)

    scale = cam_height * 1.0 / rect.shape[0]
    dim = (int(rect.shape[1] * scale), int(rect.shape[0] * scale))
    rect = cv2.resize(rect, dim)
    
    # CONCATENATING
    cam_text_frame = np.hstack((img, rect))
    h, w, _ = cam_text_frame.shape
    scale = key_width * 1.0 / w
    cam_text_frame = cv2.resize(cam_text_frame, (int(scale * w), int(scale*h)))
    

    display_frame = np.vstack((cam_text_frame, keyboard))

    print("cursor x, y: (%d, %d)" % (cursor_x, cursor_y)) 
    cv2.rectangle(display_frame,(cursor_x, cursor_y),(cursor_x+cursor_len,cursor_y+cursor_len),(0,255,255),2)
    # DISPLAY TO WINDOW
    cv2.imshow(window_name, display_frame)

    # WAIT FOR KEYBOARD
    k = cv2.waitKey(30) & 0xff
    # code, r, c, keydict, text, isShift
    if k == 27:
        break
    if k == 119 or k == 87: # w - UP
        user_keyrow, user_keycol, text, isShift = updateDirection(2, user_keyrow, user_keycol, keydict, text, isShift)
    if k == 65 or k == 97: # a - LEFT
        user_keyrow, user_keycol, text, isShift = updateDirection(0, user_keyrow, user_keycol, keydict, text, isShift)
    if k == 84 or k == 115: # s - DOWN
        user_keyrow, user_keycol, text, isShift = updateDirection(3, user_keyrow, user_keycol, keydict, text, isShift)
    if k == 68 or k == 100: # d - RIGHT
        user_keyrow, user_keycol, text, isShift = updateDirection(1, user_keyrow, user_keycol, keydict, text, isShift)
    if k == 32: # d - SPACE
        text, isShift = enterCharacter(user_keyrow, user_keycol, keydict, text, isShift)
    else:
        user_keyrow, user_keycol, text, isShift = updateDirection(input_dir, user_keyrow, user_keycol, keydict, text, isShift)
    cursor_x, cursor_y = updateCursor(user_keyrow, user_keycol)

    if (input_dir != None):
        input_dir = None

cap.release()
cv2.destroyAllWindows()

