#------------imports---------------#

import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils
from keras.models import load_model


#--------------capConts-----------------#

cap = cv2.VideoCapture(0)
brightness = 180
threshold = 0.75         
font = cv2.FONT_HERSHEY_SIMPLEX 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(10, brightness)
model= load_model('model_trained.h5')

#-------------landmark----------------#

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#--------------status-----------------#

sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

#---------------compute--------------#

def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

#---------------blink----------------#

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

    #------Active,sleepy,Drowsy------#

	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0

#--------------gray----------------#

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

#------------equalize--------------#

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

#---------preprocessing------------#

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

#----------getCalssName-------------#

def getCalssName(classNo):
    if   classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1: 
        return 'Speed Limit 30 km/h'
    elif classNo == 2: 
        return 'Speed Limit 50 km/h'
    elif classNo == 3: 
        return 'Speed Limit 60 km/h'
    elif classNo == 4: 
        return 'Speed Limit 70 km/h'
    elif classNo == 5: 
        return 'Speed Limit 80 km/h'
    elif classNo == 6: 
        return 'End of Speed Limit Less Then 80 Km/h'
    elif classNo == 7: 
        return 'Speed Limit 100 km/h'
    elif classNo == 8: 
        return 'Speed Limit 120 km/h'
    elif classNo == 9: 
        return 'No passing'
    elif classNo == 10: 
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: 
        return 'Right-of-way at the next intersection'
    elif classNo == 12: 
        return 'Priority road'
    elif classNo == 13: 
        return 'Yield'
    elif classNo == 14: 
        return 'Stop'
    elif classNo == 15: 
        return 'No vechiles'
    elif classNo == 16: 
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: 
        return 'No entry'
    elif classNo == 18: 
        return 'General caution'
    elif classNo == 19: 
        return 'Dangerous curve to the left'
    elif classNo == 20: 
        return 'Dangerous curve to the right'
    elif classNo == 21: 
        return 'Double curve'
    elif classNo == 22: 
        return 'Bumpy road'
    elif classNo == 23: 
        return 'Slippery road'
    elif classNo == 24: 
        return 'Road narrows on the right'
    elif classNo == 25: 
        return 'Road work'
    elif classNo == 26: 
        return 'Traffic signals'
    elif classNo == 27: 
        return 'Pedestrians'
    elif classNo == 28: 
        return 'Children crossing'
    elif classNo == 29: 
        return 'Bicycles crossing'
    elif classNo == 30: 
        return 'Beware of ice/snow'
    elif classNo == 31: 
        return 'Wild animals crossing'
    elif classNo == 32: 
        return 'End of all speed and passing limits'
    elif classNo == 33: 
        return 'Turn right ahead'
    elif classNo == 34: 
        return 'Turn left ahead'
    elif classNo == 35: 
        return 'Ahead only'
    elif classNo == 36: 
        return 'Go straight or right'
    elif classNo == 37: 
        return 'Go straight or left'
    elif classNo == 38: 
        return 'Keep right'
    elif classNo == 39: 
        return 'Keep left'
    elif classNo == 40: 
        return 'Roundabout mandatory'
    elif classNo == 41: 
        return 'End of no passing'
    elif classNo == 42: 
        return 'End of no passing by vechiles over 3.5 metric tons'

#---------while----------#

while True:
    #run_bot()
    ret, frame = cap.read()

    gray = grayscale(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36],landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43],landmarks[44], landmarks[47], landmarks[46], landmarks[45])
     
        if(left_blink==0 or right_blink==0):
        	sleep+=1
        	drowsy=0
        	active=0
        	if(sleep>6):
        		status="Sleeping"
        		color = (255,0,0)

        elif(left_blink==1 or right_blink==1):
        	sleep=0
        	active=0
        	drowsy+=1
        	if(drowsy>6):
        		status="Drowsy"
        		color = (0,0,255)

        else:
        	drowsy=0
        	sleep=0
        	active+=1
        	if(active>6):
        		status="Awake"
        		color = (0,255,0)
        	
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
        	(x,y) = landmarks[n]
        	cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

    #-------------red----------------#

    lower_red = np.array([0, 50, 120])
    upper_red = np.array([10, 255, 255])

    #------------yellow--------------#

    lower_yellow = np.array([25, 70, 120])
    upper_yellow = np.array([30, 255, 255])
    
    #------------green---------------#

    low_green = np.array([40, 70, 80])
    high_green = np.array([70, 255, 255])
  

    #------------mask----------------#

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, low_green, high_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #------------result--------------#

    result_green = cv2.bitwise_and(frame, frame, mask=mask_green)
    result_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    result_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)

    #------------cnts----------------#

    cnts1 = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)
    cnts2 = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    cnts3 = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)
    

    #------------ColorRcg-----------------#

    for c in cnts1:
         area = cv2.contourArea(c)
         if area > 5000:


             cv2.drawContours(frame,[c],-1,(0,0,255), 3)

             M = cv2.moments(c)

             cx = int(M["m10"]/ M["m00"])
             cy = int(M["m01"]/ M["m00"])

             cv2.circle(frame,(cx,cy),7,(255,255,255),-1)
             cv2.putText(frame, "RED", (cx-20,cy-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 3)


    for c in cnts2:
         area = cv2.contourArea(c)
         if area > 5000:


             cv2.drawContours(frame,[c],-1,(0,255,0), 3)

             M = cv2.moments(c)

             cx = int(M["m10"]/ M["m00"])
             cy = int(M["m01"]/ M["m00"])

             cv2.circle(frame,(cx,cy),7,(255,255,255),-1)
             cv2.putText(frame, "GREEN", (cx-20,cy-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), 3)


    for c in cnts3:
         area = cv2.contourArea(c)
         if area > 5000:


             cv2.drawContours(frame,[c],-1,(0,255,255), 3)

             M = cv2.moments(c)

             cx = int(M["m10"]/ M["m00"])
             cy = int(M["m01"]/ M["m00"])

             cv2.circle(frame,(cx,cy),7,(255,255,255),-1)
             cv2.putText(frame, "YELLOW", (cx-20,cy-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,255), 3)


    img = np.asarray(frame)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    #cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(frame, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)



    predictions = model.predict(img)
    probabilityValue =np.amax(predictions)
    #print(predictions.shape)
    classIndex = np.argmax(predictions)
    #print(classIndex)
    #print(probabilityValue)
    
    if probabilityValue > threshold:
        #print(getCalssName(classIndex))
        cv2.putText(frame,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    #------------show----------------#

    cv2.imshow('monitor', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()