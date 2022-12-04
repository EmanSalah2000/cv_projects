import cv2
from pyzbar.pyzbar import decode


cap=cv2.VideoCapture(0)

while True:
    state ,frame=cap.read()

    qr = decode(frame)
    if len(qr)>0:
       for code in qr:
           print(code.data.decode("UTF-8"))

    cv2.imshow("img",frame)

    if cv2.waitKey(60) and 0xff == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
