import cv2
import numpy as np

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

# Initialize webcam capture
cap = cv2.VideoCapture(0)
overlay_img = cv2.imread(r"C:/Users/prash/Desktop/SI2/Untitled design (3).jpg")

# Set video frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize overlay image to match the frame size
    overlay_resized = cv2.resize(overlay_img, (frame.shape[1], frame.shape[0]))
    
    # Create a masked region by bitwise AND operation
    img_region = cv2.bitwise_and(frame, overlay_resized)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold
    white_threshold = 50  # Adjust this threshold as needed
    _, binary_rail = cv2.threshold(gray_image, white_threshold, 255, cv2.THRESH_BINARY)
    
    # Convert binary image to BGR for visualization
    binary_with_line = cv2.cvtColor(binary_rail, cv2.COLOR_GRAY2BGR)
    
    # Define line coordinates
    x1, y1, x2, y2 = 235, 200, 395, 200
    
    # Draw initial green line
    cv2.line(binary_with_line, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Get points on the line using Bresenham's algorithm
    line_points = bresenham_line(x1, y1, x2, y2)
    
    # Check for obstacles along the line
    obstacle_detected = False
    for x, y in line_points:
        if binary_rail[y, x] == 0:
            obstacle_detected = True
            break
    
    # If an obstacle is detected, change the line to red
    if obstacle_detected:
        print("Obstacle detected!")
        cv2.line(binary_with_line, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        cv2.line(binary_with_line, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the resulting frames
    cv2.imshow('Overlay', gray_image)
    cv2.imshow('Binary with Line', binary_with_line)
    
    # Break the loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
