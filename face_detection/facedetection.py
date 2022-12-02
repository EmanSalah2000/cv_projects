# face detection


import cv2

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

stream = cv2.VideoCapture(0)

while True:
    state , Frame= stream.read()

    gray_frame=cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)
    # frame , scale , features 
    faces = face_detect.detectMultiScale(gray_frame ,1.3,5)

 
    #  faces =  x y  width height
    for (x,y,w,h) in faces:
        cv2.rectangle(Frame , (x,y),(x+w,y+h) ,(0,255,0),2)
    
    cv2.imshow("live_stream" , Frame)

    if cv2.waitKey (50) & 0xff == ord("q"):
        break
stream.release()
cv2.destroyAllWindows()

