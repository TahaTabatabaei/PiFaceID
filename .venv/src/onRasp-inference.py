# import tflite_runtime.interpreter as tflite
import tensorflow as tf
import os
import time
import numpy as np
import sys

def predict(model, samples):
    input = model.get_input_details()
    output = model.get_output_details()

    # print(f"input: {input}")
    # print(f"output: {output}")

    # Test the model on random input data.
    input_shape = input[0]['shape']
    # print(f"shape: {input_shape}")

    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    outputs = []
    for sample in samples:
        # print(f"sample shape: {sample.shape}")
        input_data = sample.reshape(input_shape)
        print(f"in  shape: {input_data.shape}")
        
        #input_data = np.expand_dims(input_data, axis=0)
        st = time.time()
        model.set_tensor(np.float32(input[0]['index']), input_data.astype('float32'))
        model.invoke()
        end = time.time()
        print(f"duration: {end-st} s") 

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = model.get_tensor(output[0]['index'])
        #print(output_data)
        outputs.append(output_data)
    ret = np.stack(outputs)
    return ret

from PIL import Image
def read_image(filename):

    # load image from file
    image = Image.open(filename)

    # convert to RGB, if needed
    image = image.convert('RGB')

    # convert to array
    pixels = np.asarray(image)

    return pixels


base_dir = "src/keras/content/keras-facenet"
sys.path.append(base_dir + '/code/')

tf_model_dir = base_dir + '/model/20180402-114759/'
npy_weights_dir = base_dir + '/model/keras/npy_weights/'
weights_dir = base_dir + '/model/keras/weights/'
model_dir = base_dir + '/model/keras/model/'

weights_filename = 'facenet_keras_weights.h5'
model_filename = 'facenet_keras.h5'
model_filename_new = "facenet_keras.keras"
model_filename_tfl = "facenet_keras.tflite"
tfl_default_optimized_v1 = "tf1_default_optimized_v1.tflite"


taha1 = read_image("src/dataset/myImages/crop/train/taha/IMG_1063-cropped.jpg")
taha2 = read_image("src/dataset/myImages/crop/train/taha/IMG_0604-cropped.jpg")
jerry1 = read_image("src/dataset/myImages/crop/train/jerry_seinfeld/httpgraphicsnytimescomimagessectionmoviesfilmographyWireImagejpg-cropped.jpg")
jerry2 = read_image("src/dataset/myImages/crop/train/jerry_seinfeld/httpikinjaimgcomgawkermediaimageuploadsWmIuhdsrcedidjpgjpg-cropped.jpg")

imgs = [taha1,taha2,jerry1,jerry2]

# preprocess
input_imgs = []
for i in imgs:
    input_imgs.append(i.reshape((1, 160, 160, 3)))

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=os.path.join(model_dir, model_filename_tfl))
# interpreter = tflite.Interpreter(model_path=os.path.join(model_dir, model_filename_tfl))

interpreter.allocate_tensors()

# Create arrays to store the results
_predictions = predict(interpreter, input_imgs)

print("distance taha1 vs taha2", np.linalg.norm(_predictions[0, :] - _predictions[1, :]))
print("distance jerry1 vs jerry2", np.linalg.norm(_predictions[2, :] - _predictions[3, :]))
print("distance jerry1 vs taha1", np.linalg.norm(_predictions[2, :] - _predictions[0, :]))

interpreter2 = tf.lite.Interpreter(model_path=os.path.join(model_dir, tfl_default_optimized_v1))
# interpreter2 = tflite.Interpreter(model_path=os.path.join(model_dir, tfl_default_optimized_v1))

interpreter2.allocate_tensors()

# Create arrays to store the results
_predictions2 = predict(interpreter2, input_imgs)

print("distance taha1 vs taha2", np.linalg.norm(_predictions2[0, :] - _predictions2[1, :]))
print("distance jerry1 vs jerry2", np.linalg.norm(_predictions2[2, :] - _predictions2[3, :]))
print("distance jerry1 vs taha1", np.linalg.norm(_predictions2[2, :] - _predictions2[0, :]))