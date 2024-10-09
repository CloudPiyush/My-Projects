import cv2
from random import randrange

# Paths to the image, video, and classifier files
image_path = "D:/Project/Car_And_Pedestrian_Tracker/Images/Image3.jpg"
video_path = "D:/Project/Car_And_Pedestrian_Tracker/Videos/Video5.mp4"


car_classifier_file = "D:/Project/Car_And_Pedestrian_Tracker/cars.xml"
pedestrian_classifier_file = "D:/Project/Car_And_Pedestrian_Tracker/haarcascade_fullbody.xml"

# Load the car and pedestrian classifiers
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

# Check if the classifiers are loaded correctly
if car_tracker.empty():
    raise IOError(f"Failed to load car classifier from {car_classifier_file}")
if pedestrian_tracker.empty():
    raise IOError(f"Failed to load pedestrian classifier from {pedestrian_classifier_file}")

# Load the video
video = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = video.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

# Set the width and height for the video frames
width, height = 900, 600

# Run forever to process video frames
while True:
    # Read the current frame
    read_successful, frame = video.read()

    if not read_successful:
        break

    # Resize the frame
    resize_frame = cv2.resize(frame, (width, height))
    
    # Convert frame to grayscale
    gray_scale = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars and pedestrians in the frame
    cars = car_tracker.detectMultiScale(gray_scale)
    pedestrians = pedestrian_tracker.detectMultiScale(gray_scale)

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(resize_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Draw rectangles around detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(resize_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Display the frame with detected objects
    cv2.imshow('Car and Pedestrian Tracker System', resize_frame)

    # Wait for key press and check if 'Q' or 'q' is pressed to exit
    key = cv2.waitKey(int(1000 / fps))
    if key == 81 or key == 113:
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

print("Code completed.")
