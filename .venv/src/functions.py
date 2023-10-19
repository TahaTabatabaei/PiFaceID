#TODO: deepface lib detectors should be checked

# function for face detection with mtcnn
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

from os import listdir
from os.path import isdir
import os

from skimage.util import random_noise
from skimage.transform import rotate
from skimage.filters import gaussian

filename = "image.JPG"

def flip(image):
    # random flip
    image = np.fliplr(image)
    return image

def rotate_with_angle(image, angle):
    # random rotation
    image = rotate(image, angle)
    return image

def add_noise(image):
    # random noise
    image = random_noise(image, mode='gaussian', var=0.05)
    return image

def blur(image):
    # random blur
    image = gaussian(image, sigma=3)
    return image


 
# extract a single face from a given photograph
def extract_face(filename, cropDir, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)

    # convert to RGB, if needed
    image = image.convert('RGB')

    # convert to array
    pixels = np.asarray(image)

    # create the detector, using default weights
    detector = MTCNN()

    # detect faces in the image
    results = detector.detect_faces(pixels)

    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']

    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = pixels[y1:y2, x1:x2]

    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)

    # save croped images
    image.save(cropDir[:-4]+"-cropped.jpg")
    print(cropDir[:-4]+"-cropped.jpg")


    # augment
    transformed_image = [face]

    flipped_image = flip(face)
    transformed_image.append(flipped_image)
    flipped_image = Image.fromarray(flipped_image)
    flipped_image = flipped_image.resize(required_size)
    flipped_image.save(cropDir[:-4]+"-cf.jpg")

    rotated_imagel = rotate_with_angle(face, 22)
    transformed_image.append(rotated_imagel)
    rotated_imagel = Image.fromarray((rotated_imagel* 255).astype(np.uint8))
    rotated_imagel = rotated_imagel.resize(required_size)
    rotated_imagel.save(cropDir[:-4]+"-crl.jpg")

    rotated_imager = rotate_with_angle(face, -22)
    transformed_image.append(rotated_imager)
    rotated_imager = Image.fromarray((rotated_imager* 255).astype(np.uint8))
    rotated_imager = rotated_imager.resize(required_size)
    rotated_imager.save(cropDir[:-4]+"-crr.jpg")

    names = ["-cn.jpg", "-cfn.jpg", "-crln.jpg", "-crrn.jpg"]
    names2 = ["-cb.jpg", "-cfb.jpg", "-crlb.jpg", "-crrb.jpg"]
    for img, name, name2 in zip(transformed_image, names, names2):
        nosisy_image = add_noise(img)
        nosisy_image = Image.fromarray((nosisy_image* 255).astype(np.uint8))
        nosisy_image = nosisy_image.resize(required_size)
        nosisy_image.save(cropDir[:-4]+name)

        blurred_image = blur(img)
        blurred_image = Image.fromarray((blurred_image* 255).astype(np.uint8))
        blurred_image = blurred_image.resize(required_size)
        blurred_image.save(cropDir[:-4]+name2)
        

    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory,cropDir):
    faces = list()

    # enumerate files
    for filename in listdir(directory):

        # path
        path = directory + filename
        cdir = cropDir + filename

        # get face
        face = extract_face(path, cdir)
        
        # store
        faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory,cropDir):
    X, y = list(), list()  

    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        cdir = cropDir + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
    
        if not os.path.exists(cdir):    
            os.mkdir(cdir)
            
        faces = load_faces(path, cdir)
        # load all faces in the subdirectory
    
        # create labels
        labels = [subdir for _ in range(len(faces))]
        
        # summarize progress
        # print('>loaded %d examples for class: %s' % (len(faces), subdir))
        
        # store
        X.extend(faces)
        y.extend(labels)
    
    return np.asarray(X), np.asarray(y)

from tensorflow.compat.v1 import gfile  # Import gfile from TensorFlow v1 for backward compatibility

def load_and_convert(model_path, tflite_model_path):

    # step 1: Load model
    model = tf.saved_model.load(model_path)

    # Step 2: Convert the TensorFlow model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()

    # Step 3: Save the TensorFlow Lite model to a .tflite file
    #  tflite_model_path = something like 'converted_model.tflite'  
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TensorFlow Lite model saved to: {tflite_model_path}")


def graph_to_savedModel(frozen_graph_path, saved_model_path):
    # Path to frozen graph file
    # frozen_graph_path = "/path/to/frozen_graph.pb"
        
    # Disable eager execution
    tf.compat.v1.disable_eager_execution()

    # Load frozen graph
    with tf.io.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Create graph from graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # Path to SavedModel directory
    # saved_model_path = "/path/to/saved_model"

    # Create SavedModel builder
    if os.path.exists(saved_model_path):
        print("\n ******** saved model is already exists ******** \n")
        return
    else:
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_path)

        # Define input and output signatures
        inputs = {
            "input:0": tf.compat.v1.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("input:0"))
        }

        outputs = {
            "embeddings:0": tf.compat.v1.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("embeddings:0"))
        }

        # Define signature definition
        signature_def = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        # Add graph to SavedModel
        builder.add_meta_graph_and_variables(
            sess=tf.compat.v1.Session(),
            tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
            }
        )

        # Save SavedModel
        builder.save()
        print("\n ******** created sucssesfully ******** \n")
        # Re-enable eager execution (if needed)
        # tf.compat.v1.enable_eager_execution()



import csv

def save_to_csv(file_path, data, labels):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([labels] + data.tolist())



from scipy.spatial.distance import cosine

def find_max_cosine_similarity(file_path, new_array):
    max_similarity = -1  # Initialize to a value lower than possible cosine similarity
    most_similar_label = None

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            label = row[0]
            stored_array = np.array(list(map(float, row[1:])))
            similarity = 1 - cosine(new_array, stored_array)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_label = label

    return most_similar_label, max_similarity





