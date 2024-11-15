import cv2
import numpy as np

def detect_and_extract_screen(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in the image.")
        return None

    # Find the largest contour (assuming it's the screen)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If the polygon has 4 vertices, we assume it's a rectangle (screen)
    if len(approx) == 4:
        screen_contour = approx
    else:
        # If not a rectangle, find the bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        screen_contour = np.int0(box)

    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Define the destination points for the perspective transform
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # Order the screen contour points
    screen_contour = order_points(screen_contour)

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(screen_contour, dst_pts)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped

def order_points(pts):
    # Ensure pts is a numpy array
    pts = np.array(pts)
    
    # Reshape pts to ensure it's 2D
    pts = pts.reshape(-1, 2)

    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the points
    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# Use the function
screen_image = detect_and_extract_screen('1.png')

if screen_image is not None:
    # Save or display the result
    cv2.imwrite('extracted_screen.jpg', screen_image)
    cv2.imshow('Extracted Screen', screen_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to extract screen from the image.")

