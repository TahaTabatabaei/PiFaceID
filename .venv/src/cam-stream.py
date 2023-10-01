from picamera import PiCamera
from picamera.array import PiRGBArray
import time
from PIL import Image, ImageDraw
import cv2
from mtcnn.mtcnn import MTCNN

# Initialize the PiCamera
camera = PiCamera()
camera.resolution = (640, 480)  # Adjust the resolution as needed
raw_capture = PiRGBArray(camera)

# Allow the camera to warm up
time.sleep(2)

# Initialize the MTCNN detector
mtcnn_detector = MTCNN()

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # Extract the BGR frame
    bgr_frame = frame.array

    # Detect faces using MTCNN
    faces = mtcnn_detector.detect_faces(bgr_frame)

    # Create a Pillow Image from the BGR frame
    pil_image = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, width, height = face['box']
        draw.rectangle([(x, y), (x + width, y + height)], outline="green", width=2)

    # Convert the Pillow Image back to a BGR frame for PiCamera display
    final_frame = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)

    # Display the final image on the PiCamera video stream
    cv2.imshow('Face Detection', final_frame)

    # Clear the stream for the next frame
    raw_capture.truncate(0)

    # Break the loop when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Close the PiCamera video stream
cv2.destroyAllWindows()

# Release the camera
camera.close()
