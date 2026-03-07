A deep learning powered REST API that detects potato leaf diseases from images using TensorFlow and FastAPI.

This project exposes a machine learning model through a production-ready API that can be integrated with web or mobile applications.

Public endpoint:

https://potato-disease-api-kbjy.onrender.com

Health check:

GET /

Prediction endpoint:

POST /predict

Model

The model is a Convolutional Neural Network trained to classify potato leaf images into three categories:

Potato Early Blight
Potato Late Blight
Healthy

Tech Stack

Python
TensorFlow
FastAPI
Uvicorn
Pillow
NumPy
Matplotlib

Run the API Locally

Start the FastAPI server:
uvicorn main:app --reload
The API will be available at:
http://127.0.0.1:8000
Interactive documentation:
http://127.0.0.1:8000/docs


FastAPI Deployment

The API is deployed on Render and automatically redeploys when changes are pushed to GitHub.
Deployment stack:

GitHub (source repository)
Render (hosting)
FastAPI + Uvicorn server

Frontend Deployment
UI: React app
Deployed on Vercel: https://potato-diseases-react-app1.vercel.app
