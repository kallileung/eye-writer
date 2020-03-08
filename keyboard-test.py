import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale 
fontScale = 5
color = (0,0,0)
thickness = 2 # font 2px

window_name = 'Image'
text = 'some text'

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


isShift = False

keyboard = None

while 1:
    if isShift:
        keyboard = KEYBOARD_SHIFT
    else:
        keyboard = KEYBOARD_NORMAL

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
    org = (30, 30)
    rect = cv2.putText(rect, text, org, font, fontScale,  
                 color, thickness, cv2.LINE_AA, False)

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

    # DISPLAY TO WINDOW
    cv2.imshow(window_name, display_frame)

    # WAIT FOR KEYBOARD
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
