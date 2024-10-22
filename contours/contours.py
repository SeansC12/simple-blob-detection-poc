import cv2
import numpy as np

# Step 1: Preprocess - Thresholding (Normal for white capsules on black background)
image = cv2.imread('contours.jpeg')

lower = np.array([0, 0, 0])
upper = np.array([55, 55, 55])

brightness = -100
contrast = 100
image = np.int16(image)
image = image * (contrast/127+1) - contrast + brightness
image = np.clip(image, 0, 255)
image = np.uint8(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Since capsules are white, use THRESH_BINARY
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 2: Morphological Operations (Opening - Erosion followed by Dilation)
kernel = np.ones((3, 3), np.uint8)
# Erosion - shrink the capsules to separate them
eroded = cv2.erode(thresh, kernel, iterations=2)
# Dilation - restore capsules to their original size
dilated = cv2.dilate(eroded, kernel, iterations=3)

# Step 3: Distance Transform
dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
# Normalize distance transform for visualization
dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

# Thresholding the distance transform to get the foreground (capsule centers)
_, sure_fg = cv2.threshold(dist_transform, 0.5, 1.0, 0)
sure_fg = np.uint8(sure_fg)

# Step 4: Watershed Algorithm
# Finding unknown region (background minus sure foreground)
unknown = cv2.subtract(dilated, sure_fg)

# Label the sure foreground (capsules)
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so that the background becomes 1 instead of 0
markers = markers + 1

# Mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply the Watershed algorithm
markers = cv2.watershed(image, markers)

# Step 5: Draw Contours for Visualization
image[markers == -1] = [255, 0, 0]  # Mark the boundary with red

cv2.imshow("contours", image)

thresh = cv2.inRange(image, lower, upper)

# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Display the result
cv2.imshow('Separated Capsules', morph)

# Step 3: Contour Detection
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Calculate Capsule Area (using Contours)
for contour in contours:
    area = cv2.contourArea(contour)
    # Optionally, get rotated bounding box if needed
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    print("Capsule area:", area)

print("Total area of image:", image.shape[0] * image.shape[1])

# Display result
cv2.imshow('Capsule Segmentation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# image = cv2.imread("contours.jpeg")

# brightness = -100
# contrast = 100
# image = np.int16(image)
# image = image * (contrast/127+1) - contrast + brightness
# image = np.clip(image, 0, 255)
# image = np.uint8(image)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
 
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
 
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
 
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)

# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
 
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
 
# # Now, mark the region of unknown with zero
# markers[unknown == 255] = 0

# markers = cv2.watershed(image,markers)
# image[markers == -1] = [255, 0, 0]

# cv2.imshow("image", image)

# cv2.waitKey(0)