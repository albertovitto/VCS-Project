import cv2 as cv


# https://stackoverflow.com/questions/60065707/cant-use-sift-in-opencv-v4-20/60066177#60066177

# https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/
# https://aishack.in/tutorials/sift-scale-invariant-feature-transform-features/
# https://stackoverflow.com/questions/29133085/what-are-keypoints-in-image-processing
image = cv.imread("home.jpg")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d_SIFT.create()
keyPoints = sift.detect(image, None)
keypoints, descriptor = sift.compute(gray_image, keyPoints)

output = cv.drawKeypoints(image, keyPoints, None)

cv.imshow("FEATURES DETECTED", output)
cv.imshow("NORMAL", image)

cv.waitKey(0)
cv.destroyAllWindows()
