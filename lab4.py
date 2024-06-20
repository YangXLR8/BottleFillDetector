import cv2                                                                                      
import numpy as np                                                                              
import os                                                                                      
from sklearn.linear_model import LinearRegression                                              
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline 
import matplotlib.pyplot as plt                                                               
import imutils                                                                                

def main():
    labeled_dirs = {                                                                           
        '50mL': 'data-lab04/50mL',
        '100mL': 'data-lab04/100mL',
        '150mL': 'data-lab04/150mL',
        '200mL': 'data-lab04/200mL',
        '250mL': 'data-lab04/250mL',
        '300mL': 'data-lab04/300mL',
        '350mL': 'data-lab04/350mL',
    }
    
    
    guess_dir_base = 'data-lab04/guess'                                                        
    subfolders = ['A', 'B', 'C']                                                             

    features, volumes = [], []                                                                 
    
    for volume, directory in labeled_dirs.items():                                             
        data, _ = load_and_process_images(directory)                                            
        features.extend(data)                                                                  
        volumes.extend([int(volume[:-2])] * len(data))                                         

    model = make_pipeline(StandardScaler(), LinearRegression())                                
    model.fit(features, volumes)                                                             

    display_results(guess_dir_base, subfolders, model)

def load_and_process_images(directory):
    
    images = []                                                        
    for filename in os.listdir(directory):                                                     
        if filename.lower().endswith('.jpg'):                                       
            path = os.path.join(directory, filename)                                         
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:                                         
                image = cv2.resize(image, (500, 500))                                         
                images.append(image)
    if images:
        features = [process_image(img) for img in images]                                     
        return features, images
    else:
        return [], None     
    
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                           
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)                                                
    
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)                       
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))                                 
                                                                                               
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)                               
    
    contours = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
                                                                                            
    contours = imutils.grab_contours(contours)                                                 
    if contours:            
        largest_contour = max(contours, key=cv2.contourArea)                                    
        x, y, w, h = cv2.boundingRect(largest_contour)                                        
        return w * h, h                                                                         
    
    else:                                                                                      
        return 0, 0                                                                            
                                                                   


def display_results(guess_dir_base, subfolders, model):
    num_subfolders = 3                                                                          

    fig, axs = plt.subplots(1, num_subfolders, figsize=(5 * num_subfolders, 5))                
                                                                                               
    for ax, subfolder in zip(axs, subfolders):                                                 
        guess_dir = os.path.join(guess_dir_base, subfolder)                                    
        guess_features, representative_images = load_and_process_images(guess_dir)

        if representative_images is not None:                                                  
            predicted_volumes = model.predict(guess_features)      
            average_volume = np.mean(predicted_volumes)                                       
            ax.imshow(cv2.cvtColor(representative_images[0], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Subfolder {subfolder}\nAvg. Volume: {average_volume:.2f}mL")
            ax.axis('off')

        else:
            ax.axis('off')
            ax.set_title(f"No images in Subfolder {subfolder}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('results.png')
    plt.show()


if __name__ == "__main__":
    main()
