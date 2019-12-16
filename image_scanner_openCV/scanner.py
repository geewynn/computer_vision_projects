from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils

img = cv2.imread('omr_test_01.png')
#img = cv2.imread('cert.jpg')
ratio = img.shape[0]/500.0
original = img.copy()
img = imutils.resize(img, height=500)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

print("STEP 1: Edge Detection")
# cv2.imshow("Image", img)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, 
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key= cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

print("STEP 2: Find contours of paper")
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()  

warped = four_point_transform(original, screenCnt.reshape(4, 2) * ratio)
blackwhite = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = 'gaussian')
# warped = (warped> T).astype('uint8')*255


print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(original, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.imshow('black and white', imutils.resize(blackwhite, height=650))
#cv2.imshow("Color", imutils.resize(color, height = 650))
cv2.waitKey(0)