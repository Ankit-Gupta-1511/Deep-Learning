from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


data = loadmat('./data/frey_rawface.mat')
faces = data['ff'].T

def show_faces_grid(data, rows=5, cols=5):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(data):
            face_image = data[i].reshape(28, 20)
            ax.imshow(face_image, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_faces_grid(faces, 5, 5)

