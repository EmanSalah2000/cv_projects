# optical character recognition
import cv2
import pytesseract



pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# read the image
img=cv2.imread("images\\images_1.jpg")

extract_text = pytesseract.image_to_string(img)

print(extract_text)
cv2.imshow(extract_text,img)
