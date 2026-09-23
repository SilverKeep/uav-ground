import cv2
import numpy as np

"""
Isolate and show unique colors in an image in separate frames
Too sensitive--need to group unique colors within some range (isolate_color allows it, make the change in unique_colors)
"""

img = cv2.imread("in.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

height, width, channels = img.shape
total_pixels = height * width

def isolate_color(lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    isolated = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("single color: ", isolated)
    cv2.waitKey(0)

unique_colors, pixel_counts = np.unique(hsv.reshape(-1, 3), axis=0, return_counts=True)
for color, count in zip(unique_colors, pixel_counts):
    if count < 0.1 * total_pixels:
        continue

    isolate_color(color, color)