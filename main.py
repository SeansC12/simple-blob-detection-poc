import cv2
import numpy as np

# Load the image
image_path = "new_imgs/IMG_3224.jpg"  # Change this to the path of your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply a Gaussian blur to reduce noise and improve blob detection
blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
blurred_image = cv2.bitwise_not(blurred_image)

# Set up the blob detector with parameters tuned for pill detection
params = cv2.SimpleBlobDetector_Params()

# Filter by area: Pills are generally medium-sized objects
params.filterByArea = True
params.minArea = 300  # Adjusted for the image resolution and pill size
params.maxArea = 100000000  # Tuned to ignore overly large objects

# Filter by circularity: Pills are often round or oval
params.filterByCircularity = True
params.minCircularity = 0.1  # Set lower to accommodate oval pills

# Filter by convexity: Pills typically have a convex shape
params.filterByConvexity = True
params.minConvexity = 0.7  # Can be adjusted for different pill shapes

# Filter by inertia: Helps to detect elongated objects like capsules
params.filterByInertia = True
params.minInertiaRatio = 0.1  # Allows detection of both circular and elongated pills

# Set up the detector with the tuned parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs (pills)
keypoints = detector.detect(blurred_image)

# Draw detected blobs as red circles
# output_image = cv2.drawKeypoints(blurred_image, keypoints, np.array([]), (255, 0, 0),
#                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def draw_keypoints(img, keypoints, color=(255, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(img, (int(x), int(y)), color=color, radius=10, thickness=10) # you can change the radius and the thickness

draw_keypoints(blurred_image, keypoints, color=(0, 255, 255))

# Show the output image with detected pills
cv2.imshow("Detected Pills", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()