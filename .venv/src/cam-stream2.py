import picamera
import picamera.array
from PIL import Image, ImageDraw
from mtcnn.mtcnn import MTCNN

# Initialize the camera
with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)  # Set the resolution as needed
    camera.framerate = 30  # Set the framerate as needed

    # Create a stream for capturing video frames
    with picamera.array.PiRGBArray(camera) as stream:
        detector = MTCNN()  # Initialize the MTCNN detector

        # Capture frames from the camera
        for frame in camera.capture_continuous(stream, format="rgb"):
            image = frame.array
            image = Image.fromarray(image)

            # Detect faces in the frame
            faces = detector.detect_faces(image)

            # Draw bounding boxes around detected faces
            draw = ImageDraw.Draw(image)
            for face in faces:
                x, y, width, height = face['box']
                draw.rectangle(((x, y), (x + width, y + height)), outline="red", width=2)

            # Show the frame with bounding boxes
            image.show()

            # Clear the stream for the next frame
            stream.seek(0)
            stream.truncate()

            # Press 'q' to exit the loop
            if input("Press 'q' to quit, or Enter to continue: ") == 'q':
                break
