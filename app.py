import cv2
import streamlit as st
import numpy as np
from camera_input_live import camera_input_live

image = camera_input_live()

if image:
  st.image(image)

