import cv2
import streamlit as st
import numpy as np

device_path = "USB\VID_2BDF&PID_0284&MI_00\6&5D2DB91&0&0000"

# For webcam input:
cap = cv2.VideoCapture(device_path)

st.title("Obj Det webcam")
frame_placeholder = st.empty()
st.write(cap.isOpened())
