Overview

This project implements an SMS Spam Classification system using machine learning. 

The system is designed to classify incoming SMS messages as either "spam" or "ham" (not spam). 

The classification is performed using a trained machine learning model, which has been serialized and saved for deployment.

Features

Machine Learning Model: Utilizes a trained model to classify SMS messages.

Serialization: The trained model is saved as a .pkl file for easy deployment and usage.

Deployment Ready: Includes necessary files such as Procfile for deployment on platforms like Heroku.

Deployment:

The project is ready for deployment. 
If you're deploying on Heroku, make sure to include the Procfile in the root directory.

File Descriptions

main.py: The main Python script that loads the model and performs classification on input SMS messages.

model.pkl: The serialized machine learning model trained on SMS data.

requirements.txt: Contains all the dependencies required to run the project.

Procfile: Necessary for deploying the application on platforms like Heroku.

Model

The model used in this project was trained using scikit-learn and is capable of classifying SMS messages with high accuracy. 

The serialized model (model.pkl) can be easily loaded and used for classification.
