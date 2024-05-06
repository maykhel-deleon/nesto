import cv2
import streamlit as st
import numpy as np

# For webcam input:
cap = cv2.VideoCapture(0)

st.title("Obj Det webcam")
frame_placeholder = st.empty()
st.write(cap.isOpened())
