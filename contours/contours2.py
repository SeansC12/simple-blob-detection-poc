import cv2
import numpy as np

# Step 1: Preprocess - Thresholding (Normal for white capsules on black background)
image = cv2.imread('imgs/area_not_working.jpg')

brightness = -200
contrast = 200
image = np.int16(image)
image = image * (contrast/127+1) - contrast + brightness
image = np.clip(image, 0, 255)
image = np.uint8(image)

image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

cv2.imshow("original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 2: Morphological Operations (Opening - Erosion followed by Dilation)
kernel = np.ones((5), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=2)
dilated = cv2.dilate(eroded, kernel, iterations=3)

# Step 3: Distance Transform
dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

# Thresholding the distance transform to get the foreground (capsule centers)
_, sure_fg = cv2.threshold(dist_transform, 0.5, 1.0, 0)
sure_fg = np.uint8(sure_fg)

# Step 4: Watershed Algorithm
unknown = cv2.subtract(dilated, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # Ensure background is labeled as 1
markers[unknown == 255] = 0

# Apply the Watershed algorithm
colored_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
markers = cv2.watershed(colored_image, markers)

test_img = colored_image.copy()
test_img[markers == -1] = [255, 0, 0]
cv2.imshow("Watershed", test_img)

# Step 5: Find the area of each capsule
unique_labels = np.unique(markers)
capsule_areas = []

for label in unique_labels:
    if label > 1:  # Skip background (1) and borders (-1)
        # Create a mask for the current label
        capsule_mask = np.uint8(markers == label)
        # Calculate the area as the number of pixels in the mask
        area = cv2.countNonZero(capsule_mask)
        capsule_areas.append((label, area))

        # Optionally visualize each capsule
        mask_bgr = cv2.merge([capsule_mask * 255] * 3)
        capsule_visualization = cv2.addWeighted(colored_image, 0.8, mask_bgr, 0.2, 0)
        cv2.imshow(f'Capsule {label}', capsule_visualization)
        cv2.waitKey(0)

# Display the capsule areas
for label, area in capsule_areas:
    print(f"Capsule {label}: Area = {area} pixels")

cv2.destroyAllWindows()
