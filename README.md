<h1 align="center">Volume Prediction Using Image Processing</h1>
<h3 align="center">Laboratory Output for Computer Vision</h3>

<p align="center">
  <img width="700" src="https://github.com/YangXLR8/BottleFillDetector/blob/main/results.png" alt="cli output"/>
</p>

## DESCRIPTION

This project aims to predict the volume of liquid in containers using image processing and machine learning techniques. It uses free assets for image data and processes them to extract features for training a regression model.

## Project Structure

- `main.py`: Main script to run the volume prediction.
- `data-lab04/`: Directory containing labeled and guess images.
- `lab4comments.py/`: Main script with comments
- `00-README.txt`: Laboratory Instructions

## Methodology

1. **Data Loading**: Images are loaded from the specified directories.
2. **Image Processing**: Images are processed to extract features such as contour areas.
3. **Model Training**: A linear regression model is trained using the extracted features.
4. **Volume Prediction**: The trained model predicts the volume of new images, and the results are displayed and saved.

## Results

The `results.png` file will contain the images from the `guess` folders along with their average predicted volumes.

## Colaborators

👤 **Cantiller, Sophia Feona** - Github: [@SophiaCantiller](https://github.com/SophiaCantiller)

👤 **Sardañas, Reah Mae** - Github: [@YangXLR8](https://github.com/YangXLR8)
