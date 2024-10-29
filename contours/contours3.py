import cv2
import numpy as np

# Load the image
image_path = "imgs/9.jpg"  # Change to the appropriate path
image = cv2.imread(image_path)

# Convert image to HSV color space for better color segmentation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV color range for capsule segmentation (adjust these values as needed)
upper_hsv = np.array([0, 0, 21.6])    # Lower bound for capsule color
lower_hsv = np.array([0, 0, 0]) # Upper bound for capsule color

# Create a mask based on the defined color range
mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If contours are found, select the largest one as the capsule
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    capsule_mask = np.zeros_like(mask)
    cv2.drawContours(capsule_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Calculate the area of the capsule (non-zero pixels in the mask)
    capsule_area = cv2.countNonZero(capsule_mask)
    print("Capsule Area:", capsule_area)

    # Display the original image
    cv2.imshow("Original Image", image)

    # Display the initial mask
    cv2.imshow("Mask", mask)

    # Display the final capsule mask
    cv2.imshow("Capsule Mask", capsule_mask)

    # Wait for a key press to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No capsule found in the image.")