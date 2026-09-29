import cv2
import numpy as np

"""
Uses the simple blob detector--probably big improvements with contour detection instead
"""

paths = ["objects.jpg", "polka_dots_1.png", "polka_dots_2.jpg", "polka_dots_3.jpg"]

def detect_blobs(file):
    img = cv2.imread(file)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False

    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 50000
    params.filterByConvexity = True
    params.minConvexity = 0.6
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.minDistBetweenBlobs = 10
    params.minThreshold = 10
    params.maxThreshold = 245
    params.thresholdStep = 5
    detector = cv2.SimpleBlobDetector_create(params)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]

    all_pts = list(detector.detect(gray)) + list(detector.detect(sat)) + list(detector.detect(img))
    output = cv2.drawKeypoints(img, all_pts, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Blobs Detected", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for path in paths:
    detect_blobs(path)