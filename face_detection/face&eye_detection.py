# face detection
#haarcascade_eye.xml

import cv2

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detect=cv2.CascadeClassifier("haarcascade_eye.xml")

stream = cv2.VideoCapture(0)

while True:
    state , Frame= stream.read()

    gray_frame=cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)

    # face detection
    # frame , scale , features 
    faces = face_detect.detectMultiScale(gray_frame ,1.3,5)
    #  faces =  x y  width height
    for (x,y,w,h) in faces:
        cv2.rectangle(Frame , (x,y),(x+w,y+h) ,(0,255,0),2)
        # eye detection
        face_only = Frame[y:y+h , x:x+w]
        eyes = eye_detect.detectMultiScale(face_only ,1.3,5)

        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(face_only , (ex,ey),(ex+ew,ey+eh) ,(0,0,255),2)
            eye_x= int((ex+(ew/2))-10)
            eye_y=int((ey+(eh/2))+10)
            
            cv2.putText(face_only,"X", (eye_x,eye_y),cv2.FONT_HERSHEY_COMPLEX ,1,(0,0,255),3)
            



#show    
    cv2.imshow("live_stream" , Frame)

    if cv2.waitKey (50) & 0xff == ord("q"):
        break
stream.release()
cv2.destroyAllWindows()

