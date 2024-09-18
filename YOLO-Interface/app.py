import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import time
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Route for homepage, displays the main HTML page
@app.route("/")
def hello_world():
    return render_template('index.html')  # Renders index.html for file upload

# Route to handle image/video prediction via POST method
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:  # Check if a file is uploaded
            f = request.files['file']  # Get the uploaded file
            basepath = os.path.dirname(__file__)  # Get current directory path
            filepath = os.path.join(basepath, 'uploads', f.filename)  # Define upload path
            f.save(filepath)  # Save uploaded file to 'uploads' directory
            predict_img.imgpath = f.filename  # Save filename globally

            file_extension = f.filename.rsplit('.', 1)[1].lower()  # Get file extension

            # Image Prediction (JPG files)
            if file_extension == 'jpg':
                img = cv2.imread(filepath)  # Read image using OpenCV
                model = YOLO('best.pt')  # Load YOLO model (change 'best.pt' to your model)
                detections = model(img, save=True)  # Perform detection and save results
                return display(f.filename)  # Call display function to show results

            # Video Prediction (MP4 files)
            elif file_extension == 'mp4':
                video_path = filepath  # Path to the uploaded video
                cap = cv2.VideoCapture(video_path)  # Capture video using OpenCV

                # Get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Setup VideoWriter to save the output
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                model = YOLO('best.pt')  # Load YOLO model

                while cap.isOpened():
                    ret, frame = cap.read()  # Read each frame
                    if not ret:  # If no frame is read, exit the loop
                        break

                    results = model(frame, save=True)  # Perform YOLO detection
                    res_plotted = results[0].plot()  # Plot the detections on the frame
                    out.write(res_plotted)  # Write the processed frame to the output video

                cap.release()  # Release the video capture object
                out.release()  # Release the VideoWriter object

                return video_feed()  # Call video feed function to stream the result

    # If GET request, get the latest detected image and render it
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    image_path = folder_path + '/' + latest_subfolder + '/' + f.filename
    return render_template('index.html', image_path=image_path)  # Display image

# Route to display the detected image
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))  # Get latest folder
    directory = folder_path + '/' + latest_subfolder  # Path to latest detection
    files = os.listdir(directory)  # Get list of files in the folder
    latest_file = files[0]  # Get the first file (latest result)
    filename = os.path.join(folder_path, latest_subfolder, latest_file)  # Build path to the latest file

    file_extension = filename.rsplit('.', 1)[1].lower()  # Check file extension

    if file_extension == 'jpg':  # If it's an image, serve it
        return send_from_directory(directory, latest_file)  # Show image
    else:
        return "Invalid file format"  # If not an image, return error

# Function to stream video frames
def get_frame():
    mp4_files = 'output.mp4'  # Path to the processed video
    video = cv2.VideoCapture(mp4_files)  # Capture the video
    while True:
        success, image = video.read()  # Read frame by frame
        if not success:  # If no frame is read, exit the loop
            break
        ret, jpeg = cv2.imencode('.jpg', image)  # Convert frame to JPEG

        # Yield frame in the format required by the video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)  # Sleep to simulate real-time video

# Route to serve video feed (streaming)
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  # Stream the frames as MJPEG

# Main execution block
if __name__ == "__main__":
    # Command-line argument to specify port
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="port number")  # Default port is 5000
    args = parser.parse_args()
    model = YOLO('best.pt')  # Load YOLO model
    app.run(host="0.0.0.0", port=args.port)  # Run the Flask app on the specified port
