import imageio
import os


def store_progress_gif():

    image_folder = 'training_images'
    images = []

    for filename in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if filename.endswith('.png'):
            file_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(file_path))
    imageio.mimsave('training_output/training_progress.gif', images, fps=5)
