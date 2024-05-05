import cv2
import numpy as np
import streamlit as st
import os
import io
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter
import streamlit_webrtc as webrtc
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Define paths
PATH_TO_MODEL = './detect.tflite'
PATH_TO_LABELS = './labelmap.txt'

# Define the tflite_detect_images function
def tflite_detect_images(image, modelpath, lblpath, min_conf=0.5, txt_only=False):
    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Convert the uploaded image to numpy array
    

    # Convert the uploaded image to a PIL Image
    uploaded_image = Image.open(image)

    # Convert the PIL Image to a NumPy array
    image = np.array(uploaded_image)

    # Check the data type of the image
    #st.write(image.dtype)

    # Preprocess the image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    float_input = (input_details[0]['dtype'] == np.float32)
    if float_input:
        input_mean = 127.5
        input_std = 127.5
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform object detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

    # Display or save the image with detections
    if txt_only == False:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        st.image(image, caption="Object Detection Result", use_column_width=True)
        #plt.figure(figsize=(12, 16))
        #plt.imshow(image)
        #plt.axis('off')
        #st.pyplot()
        
    
    return 
    
def scale_resolution(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        return frame
    
# Main Streamlit app
def main():
    
  st.title('Object Detection using Webcam')
  image=cv2.VideoCapture(0)
  tflite_detect_images(image, PATH_TO_MODEL, PATH_TO_LABELS, min_conf=0.5, txt_only=False)
  
  

    

   
    

        
if __name__ == '__main__':
    main()
