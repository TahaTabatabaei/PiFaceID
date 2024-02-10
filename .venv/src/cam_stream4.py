#!/usr/bin/python3
import time
import tensorflow as tf
#import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import sys

from picamera2 import MappedArray, Picamera2, Preview
from sklearn.svm import SVC
import pickle
from PIL import Image
import csv
from scipy.spatial.distance import cosine

# This version creates a lores YUV stream, extracts the Y channel and runs the face
# detector directly on that. We use the supplied OpenGL accelerated preview window
# and delegate the face box drawing to its callback function, thereby running the
# preview at the full rate with face updates as and when they are ready.



def save_to_csv(file_path, data, labels):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([labels] + data.tolist())
        
def find_max_cosine_similarity(file_path, new_array):
    max_similarity = -1
    most_similar_label = None
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            label = row[0]
            stored_array = np.array(list(map(float,row[1:])))
            similarity = 1- cosine(new_array,stored_array)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_label =label
                
    return most_similar_label, max_similarity
    
def load_csv(file_oath):
    data_list = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data_list.append(row)
    
    nparray = p.array(map(float,data_list))
    
    return
    
def load_model(classifier_filename):
    print('Testing classifier ...')
    
    # classifier model object
    model = SVC(kernel='linear' , probability=True)
    
    
    with open(classifier_filename ,'rb') as infile:
        (model, class_names) = pickle.load(infile)
        
    print(f"Loaded classifer model from file << {classifier_filename} >> \n")
    
    return model, class_names
    
def classifiy(model, class_names, emb_array):
    
    _predictions = model.predict_proba(emb_array)
    
    best_class_indices = np.argmax(_predictions, axis=1)
    best_class_probabilities = _predictions[np.arange(len(best_class_indices)), best_class_indices]
    
    print(len(best_class_indices))
    print(best_class_indices)
    print(best_class_indices.shape)
    print(best_class_probabilities)
    print(best_class_probabilities.shape)
    print(_predictions)
    
    for i in range(len(best_class_indices)):
        print(f"{i} {class_names[best_class_indices[i]]}: {best_class_probabilities[i]}")
        
    return class_names[best_class_indices[i]]
    
def predict(model, sample):
    input = model.get_input_details()
    output = model.get_output_details()

    # print(f"input: {input}")
    # print(f"output: {output}")

    # Test the model on random input data.
    input_shape = input[0]['shape']
    #print(f"shape: {input_shape}")

    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    outputs = []
    #for sample in samples:
    #print(f"sample shape: {sample.shape}")
    
    mean ,std = sample.mean(), sample.std()
    sample = (sample - mean)/ std
    input_data = sample.reshape(input_shape)
    #print(f"in  shape: {input_data.shape}")
    
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


# draw face callback and predict routine    
def draw_faces(request):
    timestamp = time.strftime("%Y-%m-%d %X")
    
    global skipping
    global name
    
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, timestamp, origin, font, scale, colour, thickness)
        
        for i,f in enumerate(faces):
            (x, y, w, h) = [c * n // d for c, n, d in zip(faces[0], (w0, h0) * 2, (w1, h1) * 2)]
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0))
                                    

            
            if skipping < 1:
                #if i == len(faces)-1:
                skipping = 10
                
                crop = np.array(m.array[y:y+h , x:x+w])
                
                mean, std = crop.mean(), crop.std()
                crop = (crop-mean)/std
                # dsize=(160 ,160) for facenet and dsize=(112, 112) for ghostnet 
                crop = cv2.resize(crop[:,:,0:3], dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
                
                _predictions = predict(interpreter, crop)
                
                new_face = _predictions[0, :].flatten()
                
                most_similar_label, max_similarity = find_max_cosine_similarity("dataset/database_g.csv", new_face)
                
                print(f"Most similar label: {most_similar_label}, Cosine simlarity: {max_similarity}") 
                #name = classifiy(classifier, class_names, _predictions[0, :])
                
                if max_similarity > 0.40:
                    name = most_similar_label
                    print(f"name: {i}")
                else:
                    name = "Unkown"
                    print(f"name2: {i}")
            
            label = (x, y-5)
            cv2.putText(m.array, name, label, font, scale, colour, thickness)
            
               
 

if __name__ == "__main__":

    import warnings
    warnings.simplefilter("ignore")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    colour = (0, 255, 0)
    origin = (0, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    global skipping
    global name
    global lock
    lock = False
    name =""
    skipping = 0    

    # load tflite embedding model
    model_filename_tfl = "/home/taha/Documents/code/prj/face/myenv/src/model/keras/content/keras-facenet/model/keras/model/facenet_keras.tflite"
    tfl_default_optimized_v1 = "/home/taha/Documents/code/prj/face/myenv/src/model/keras/content/keras-facenet/model/keras/model/tf1_default_optimized_v1.tflite"
    ghost_basic_model = "/home/taha/Documents/code/prj/face/env/src/models/ghostFaceNet/tflites/basic_model17.tflite"
    ghost_quant_model = "/home/taha/Documents/code/prj/face/env/src/models/ghostFaceNet/tflites/quant_model17.tflite"


    interpreter = tf.lite.Interpreter(model_path=ghost_quant_model)
    interpreter.allocate_tensors()


    picam2 = Picamera2()
    picam2.start_preview(Preview.QTGL)
    #640, 480
    #320, 240
    #720, 640
    config = picam2.create_preview_configuration(main={"size": (720, 640)},
                                                 lores={"size": (640, 480), "format": "YUV420"})
    picam2.configure(config)

    (w0, h0) = picam2.stream_configuration("main")["size"]
    (w1, h1) = picam2.stream_configuration("lores")["size"]
    s1 = picam2.stream_configuration("lores")["stride"]
    faces = []
    picam2.post_callback = draw_faces
    #picam2.post_callback = draw_faces(interpreter=interpreter, classifier=classifier, class_names=class_names)

    picam2.start()
    
    start_time = time.monotonic()
    # Run for 10 seconds.
    while time.monotonic() - start_time < 10:
        skipping -=1
        print(f"skipping: {skipping}")
        buffer = picam2.capture_buffer("lores")
        grey = buffer[:s1 * h1].reshape((h1, s1))
        faces = face_detector.detectMultiScale(grey, 1.1, 3)
        
