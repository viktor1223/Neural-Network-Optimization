"""
This file will contain all plotting functions
"""
import tensorflow as tf
import keras
import time

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython.display import clear_output, display#, HTML
import numpy as np
#from plotly.offline import init_notebook_mode, iplot
def plot_image_accuracy(img, acc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    texted_image = cv2.putText(img,str(acc)+"%",(10,500), font, 5,(255,0,0))
    return texted_image
