import cv2                                                                                      # for image processing
import numpy as np                                                                              # for numerical computations 
import os                                                                                       # interacting with the file system 
from sklearn.linear_model import LinearRegression                                               # for machine learning
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline 
import matplotlib.pyplot as plt                                                                 # plotting 
import imutils                                                                                  # additional utilities for image processing



def main():
    labeled_dirs = {                                                                           # Dictionary of directories with corresponding volumes
        '50mL': 'data-lab04/50mL',
        '100mL': 'data-lab04/100mL',
        '150mL': 'data-lab04/150mL',
        '200mL': 'data-lab04/200mL',
        '250mL': 'data-lab04/250mL',
        '300mL': 'data-lab04/300mL',
        '350mL': 'data-lab04/350mL',
    }
    
    
    guess_dir_base = 'data-lab04/guess'                                                         # Base directory for guess images
    subfolders = ['A', 'B', 'C']                                                                # Subfolders in the guess directory

    features, volumes = [], []                                                                  # Lists to hold all features and corresponding volumes
    
    for volume, directory in labeled_dirs.items():                                              # Iterate over each volume and directory
        data, _ = load_and_process_images(directory)                                            # Load and process images from the directory
        features.extend(data)                                                                   # Extend the features list with data from the current directory
        volumes.extend([int(volume[:-2])] * len(data))                                          # Extend the volumes list with the volume labels repeated as many times as there are images

    model = make_pipeline(StandardScaler(), LinearRegression())                                 # Create a pipeline with standard scaler and linear regression
    model.fit(features, volumes)                                                                # Fit the model with the features and corresponding volumes

    display_results(guess_dir_base, subfolders, model)


def load_and_process_images(directory):
    
    images = []                                                        
    for filename in os.listdir(directory):                                                      #  Iterates over each file in the specified directory.
        if filename.lower().endswith('.jpg'):                                                   # finds JPG and PNG files
            path = os.path.join(directory, filename)                                            # Constructs the full path to the image file by joining the directory path with the filename.
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:                                         
                image = cv2.resize(image, (500, 500))                                           # resize the image and append it to the image list
                images.append(image)
    if images:
        features = [process_image(img) for img in images]                                       #  iterates over each image in the images list and applies the process_image function to extract features.
        return features, images
    else:
        return [], None                                                                         # Returns empty lists for both features and images



def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                              # convert input to grayscale. Grayscale images are easier to process and require less computational power.
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)                                                 # Gaussian blur to reduce noise and smooth out details. 7, 7 is the kernel used for blurring. 
    
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)                       # Thresholding is a technique used to separate objects from the background. 
                                                                                                # Pixels with intensity values above threshold (50) are set to a maximum value (255),
                                                                                                # while pixels below the threshold are set to zero.  
                                                                                                # The cv2.THRESH_BINARY_INV flag specifies that the inverse binary thresholding is applied, 
                                                                                                # meaning that foreground (above threshold) pixels become zero and background (below foreground) pixels become 255.
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))                                  # structuring element for morphological operations.
                                                                                                # Morphological operations are used to manipulate shapes in an image. 
                                                                                                # In this case, a rectangular structuring element of size 5x5 is created.
    
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)                                # applies morphological opening to the thresholded image.
                                                                                                # Morphological opening is a combination of erosion followed by dilation. 
                                                                                                # It helps remove small objects and smooth out the edges of larger objects.
    
    contours = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      # contours are detected in the opened image.
                                                                                                # Contours are curves joining continuous points along the boundary of an object. 
                                                                                                # cv2.RETR_EXTERNAL flag retrieves only the external contours, 
                                                                                                # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    
    contours = imutils.grab_contours(contours)                                                  # extracts the contours from the output of cv2.findContours
    
    if contours:            
        largest_contour = max(contours, key=cv2.contourArea)                                    # finds the largest contour based on its area (calculated by contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)                                          # creates bounding rectangle. 
        return w * h, h                                                                         # returns area of the largest contour and the height of the bounding rectangle
    
    else:                                                                                       # if no contours found,
        return 0, 0                                                                             # area and height is 0



def display_results(guess_dir_base, subfolders, model):
    num_subfolders = 3                                                                          # Calculates the number of subfolders passed to the function.

    fig, axs = plt.subplots(1, num_subfolders, figsize=(5 * num_subfolders, 5))                 # Creates a figure and a grid of subplots, where the number of subplots is equal to the number of subfolders. 
                                                                                                # The size of the figure is adjusted based on the number of subfolders to ensure proper visualization.
    for ax, subfolder in zip(axs, subfolders):                                                  # Iterates over each subplot (ax) and its corresponding subfolder name (subfolder) using the zip function.

        guess_dir = os.path.join(guess_dir_base, subfolder)                                     # Constructs the full path to the guess directory for the current subfolder.
        guess_features, representative_images = load_and_process_images(guess_dir)

        if representative_images is not None:                                                   # if representative_images were successfully loaded
            predicted_volumes = model.predict(guess_features)   # Construct                     # train the model based on the guess features
            average_volume = np.mean(predicted_volumes)                                         # Calculates the average predicted volume across all images in the current subfolder.
            ax.imshow(cv2.cvtColor(representative_images[0], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Subfolder {subfolder}\nAvg. Volume: {average_volume:.2f}mL")
            ax.axis('off')

        else:
            ax.axis('off')
            ax.set_title(f"No images in Subfolder {subfolder}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
