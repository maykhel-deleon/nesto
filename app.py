import cv2
import streamlit as st
import numpy as np

# For webcam input:
cap = cv2.VideoCapture(1)

st.tile("Obj Det webcam")
frame_placeholder = st.empty()
