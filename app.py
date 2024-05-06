import cv2
import streamlit as st
import numpy as np
from camera_input_live import camera_input_live
import os
import io
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt
from PIL import Image

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

    # Convert the uploaded image to a PIL Image
    uploaded_image = Image.open(image)

    # Convert the PIL Image to a NumPy array
    image = np.array(uploaded_image)

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
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

    detections = []

    # Prices for each detected class
    prices = {
        "bellardo_peanut": 7.95,
        "Saras-vegan-sambar-powder": 6.75,
        "Al-Alali-Pasta-Sauce-Olive-and-mushroom": 9.95,
        "Ameracan-garden-mushroom-pasta-sauce": 15.20,
        "Integrale-penne-rigate-barilla": 19.50,
        "Lucky-me-la-paz-batchoy-instant-noodle-soup": 5.25,
        "Indomie-noodles-special-chicken": 10.25,
        "nezo-table-salt": 10.45,
        "alalali-gelatin-desert-lime": 2.75,
        "Foster-clarks-creme-caramel": 2.95,
        "Tata-soulful-ragi-bites-chocos": 28.95,
        "Nestle-lion-wild": 25.99,
        "Whole-wheat-flour": 7.25,
        "alalalicornflour": 7.75,
        "nezline_oat_flakes": 7.95,
        "Safa-sugar-cubes": 4.95,
        "alicafe_classic": 24.50,
        "maggi_organic_noodle_soup": 8.99,
        "brahmin-s_chutney_powder": 3.95,
        "alshifa_blackforest_honey": 45.26,
        "nutella": 16.99,
        "sogood_almondvanilla": 13.50,
        "lipton_tea": 23.50,
        "acorsa_sliced": 9.99,
        "nanma_sunfloweroil": 10.99,
        "heinz_tikkamayonnaise": 9.99,
        "florida-s_juice": 13.50,
        "sabahoo_croissant": 8.75,
        "karachi_badampista_biscuit": 17.50,
        "bahlsen_biscuit": 15.99,
        "oreo": 15.99,
        "supreme_chocolate_popcorn": 3.99,
        "american_garden_popcorn": 12.75,
        "best_peanut": 13.99,
        "unikai_applejuice": 4.40,
        "basil_drink": 4.95,
        "mogu_drink": 4.25,
        "vimto_cordial": 10.99,
        "rasna_orange": 9.99,
        "milma_ghee": 54.99,
        "555_fried_sardines": 3.50,
        "magnolla_cheese": 12.45,
        "moonnar_coconutoil": 6.99,
        "aljameel_oil": 23.99,
        "heinz_vinegar": 8.99,
        "puck_cream_cheese": 16.25,
        "almarai_fetacheese": 6.99,
        "pinar_creamcheese": 18.99,
        "meezan_dates": 14.99,
        "oman-chips": 4.25,
        "switz_mini_cupcake": 6.99,
        "null": 0.00,

        
        # Add more classes and their prices as needed
    }

    total_price = 0

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Get class name and price
            object_name = labels[int(classes[i])]
            price = prices.get(object_name, 0)

            # Draw detection box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Add detection to list
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            # Add price to total
            total_price += price

    # Display or save the image with detections
    if txt_only == False:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Object Detection Result", use_column_width=True)

    # Print prices of detected objects
    st.write("Detected Objects and Their Prices (in AED) :")
    for detection in detections:
        st.write(f"{detection[0]}: {prices.get(detection[0], 0)}")

    # Print total price
    st.write(f"Total Price: {total_price}")

    return detections, total_price


# Main Streamlit app
def main():
    st.title('Object Detection using Image Upload')

    image = camera_input_live()

  if image:
    #st.image(image)
    min_conf_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)

        #if st.button('Start Detection'):
    tflite_detect_images(image, PATH_TO_MODEL, PATH_TO_LABELS, min_conf_threshold)
            # Do further processing with detections if needed

if __name__ == '__main__':
    main()




