# Plant Finding App

A Flask-based web application that allows users to upload images of plants and predicts their species using a pretrained model. The app utilizes the **PlantClef** pretrained model for image classification and provides predictions with high accuracy.

## Features

- Upload an image of a plant.
- Get predictions about the plant's species.
- View the top 5 possible species predictions with their probabilities.

## Requirements

This project uses a virtual environment and the following Python libraries:

- Flask
- torch
- timm
- pandas
- Pillow

You can install these dependencies using the `requirements.txt` file.

## Installation

### 1. Clone the repository

```
git clone https://github.com/erenayh/plant-finding-app.git
cd plant-finding-app
```
### 2. Set up the virtual environment

Make sure you have Python 3 installed. You can create a virtual environment by running:

```
python3 -m venv venv
```

Activate the virtual environment:

On macOS/Linux:
```
source venv/bin/activate
```
On Windows:
```
venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Download the pretrained model

You can download the pretrained model from the following link, and place it in the project directory, or specify its path in the code.

[Download Pretrained Model](https://zenodo.org/records/10848263?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk5MmJkMGMyLWM3NDEtNDZiYi04ODIzLWQ3ODE1M2JjOGIyMiIsImRhdGEiOnt9LCJyYW5kb20iOiIwYTQwOGJjY2Q0NmFiYzZjMGRlMWNkNzZmMzk3NjMxZSJ9.MKZoVFGL6jGoI3TUmnFUzTy5nrPzgf12C1RBip--ECABBsipgctxSp5U4HfRsk9rL4FfJM2Q4_vKAYb1z1Z-Zg)

### 5. Run the application

To start the Flask app, run:
```
python app.py
```

The app will be hosted at http://127.0.0.1:5000/.

### How to Use the Plant Finding App

1. **Open the App in Your Browser**  
   Run the app locally by starting the Flask server and open it in your web browser.

2. **Upload a Plant Image**  
   On the homepage, click the "Choose File" button to upload a plant image.

3. **Submit the Image for Prediction**  
   After uploading the image, click the upload button to send the image for prediction.

4. **View the Top 5 Predicted Species**  
   Once the image is processed, the top 5 predicted plant species will be displayed along with their respective probabilities.



### Notes

 
The pretrained model used in this app is trained on a large dataset that contains a wide variety of images. If a non-plant image is uploaded, the model will still try to make a prediction based on the patterns it has learned from the data. As a result, it may predict a plant species even if the image is unrelated. The app doesn't currently implement an image classification filter to specifically detect whether the uploaded image is a plant or not, which is why predictions may still occur for non-plant images.





