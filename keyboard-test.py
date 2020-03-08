import numpy as np
import cv2
import textwrap

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale 
fontScale = 2
color = (0,0,0)
thickness = 4 # font 2px

window_name = 'Writer'
text = 'some text that is a very long text that eventually becomes a paragraph. Let us see how well it can handle long paragraphs'

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

cursor_x = user_keyrow * 60 + 40
cursor_y = user_keycol * 55 + 240
cursor_len = 52

def updateCursor(r, c):
    cursor_x = r * 60 + 40
    cursor_y = c * 55 + 240
    return (cursor_x, cursor_y)

def updateDirection(code, r, c):
    if code == 1: # DOWN
        r = min(r+1, MAXROWS - 1) 
    elif code == 2: # RIGHT
        c = min(c+1, MAXCOLS - 1)
    elif code == 3: # UP
        r = max(r-1, 0)
    elif code == 4: # LEFT
        c = max(c-1, 0)
    else:
        print("Uh oh! Where are you looking?")
    return (r, c)

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
    ## CHECK INPUT HERE
    if (input_dir != None):
        updateDirection(input_dir)
        input_dir = None

    if isShift:
        keyboard = KEYBOARD_SHIFT
        keydict = keydict_shift
    else:
        keyboard = KEYBOARD_NORMAL
        keydict = keydict_reg

    # clear rectangle every loop
    rect = cv2.imread('img/empty_rect.png')
    #rect = cv2.resize()

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

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
    if k == 27:
        break
    if k == 119 or k == 87: # w - UP
        user_keyrow, user_keycol = updateDirection(3, user_keyrow, user_keycol)
    if k == 65 or k == 97: # a - LEFT
        user_keyrow, user_keycol = updateDirection(4, user_keyrow, user_keycol)
    if k == 84 or k == 115: # s - DOWN
        user_keyrow, user_keycol = updateDirection(1, user_keyrow, user_keycol)
    if k == 68 or k == 100: # d - RIGHT
        user_keyrow, user_keycol = updateDirection(2, user_keyrow, user_keycol)
    cursor_x, cursor_y = updateCursor(user_keyrow, user_keycol)

cap.release()
cv2.destroyAllWindows()

