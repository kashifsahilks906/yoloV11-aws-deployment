# YOLO-Based Object Detection API Deployment

🚀 Deployed a YOLO-based Object Detection API on AWS ECS (Fargate)

Built an end-to-end computer vision system using a custom YOLO model and deployed it as a live API using Docker and AWS.

## Project Overview

This project is a Flask web application for image-based object detection. A user uploads an image, the backend runs inference using a custom YOLO model, and the app returns an annotated image with bounding boxes and labels.

The application was packaged with Docker and deployed on AWS ECS using Fargate, which made it possible to run the model as a cloud-accessible API.

## Features

- Image upload through a simple web interface
- Object detection using a custom-trained YOLO model
- Annotated output with bounding boxes and class labels
- Clean UI for comparing original and predicted images
- Full-screen preview for inspecting detection results closely
- Dockerized application for consistent deployment
- AWS ECS Fargate deployment for scalable hosting

## Tech Stack

- Python
- Flask
- YOLO / Ultralytics
- OpenCV
- Supervision
- NumPy
- Docker
- AWS ECS
- AWS ECR

## How It Works

1. The user uploads an image from the browser.
2. The Flask backend receives the file at the `/predict` endpoint.
3. The image is passed to the YOLO model for inference.
4. Bounding boxes and labels are drawn on the image.
5. The annotated image is returned to the frontend and displayed in the predicted panel.

## Project Structure

- `index.py` - Flask app and API routes
- `inference.py` - model loading and prediction logic
- `templates/index.html` - frontend UI
- `dockerfile` - Docker image configuration
- `requirements.txt` - Python dependencies
- `best.pt` - trained YOLO model weights

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/kashifsahilks906/yoloV11-aws-deployment
cd yoloV11-aws-deployment
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python index.py
```

### 5. Open in browser

Go to:

```bash
http://127.0.0.1:8000
```

## Docker Run

### Build the image

```bash
docker build -t yolo-object-detection-app .
```

### Run the container

```bash
docker run -p 8000:8000 yolo-object-detection-app
```

Then open:

```bash
http://127.0.0.1:8000
```

## AWS Deployment Summary

The app was deployed using:

- Docker image pushed to AWS ECR
- AWS ECS service running on Fargate
- Container exposed on port `8000`
- Flask app configured to listen on `0.0.0.0`

## Deployment Issues I Faced

During deployment, I solved several practical issues that are common in real cloud ML deployments:

- Docker build errors
- ECS task startup issues
- IAM role and permission configuration
- Networking and security group setup
- Port exposure / container routing problems
- Making sure the Flask app listens on `0.0.0.0` instead of `localhost`

These issues were an important part of the learning process because they showed how model development and production deployment are different problems.

## Training Notebook

I trained the custom YOLO model in a Kaggle notebook.

### Kaggle Training Notebook Link
```text
https://www.kaggle.com/code/kashifsahil/yolo11-objectdetection-on-customdataset
```

## Results

- Upload an image
- Run detection
- View the annotated output with predicted bounding boxes
- Compare original and predicted images visually

## Key Takeaway

This project helped me learn how to move an ML model from local development to a production-ready cloud deployment.

It also showed the full pipeline from data preparation and model training to API integration, containerization, and AWS hosting.

## Future Improvements

- Add confidence threshold control in the UI
- Add support for multiple image uploads
- Show detection results in a table below the image
- Add video inference support
- Store prediction history for later review

## License

Add a license here if you want to open-source the project.