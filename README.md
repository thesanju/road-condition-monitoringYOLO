# YOLO Project
Live preview http://34.30.190.159:5000/

This project contains two main directories:
1. Training
2. Interface (for running the model)

## Training

We trained the model using only potholes dataset, you can collect as many road condition features and annotate them using [roboflow](https://roboflow.com/annotate) or any other annotation tool, make sure to use yolov8 format for annotation read more [here](https://docs.ultralytics.com/datasets/detect/) 

To train the model:

1. Navigate to the YOLO directory:
   ```
   cd yolo
   ```

2. Run the training script:
   ```
   python train.py
   ```

3. After training completes, you'll find a `best.pt` checkpoint in the `runs/train` directory.

4. Copy the `best.pt` file to the `YOLO-Interface` directory.

## Running the Model

To run the model:

1. Ensure you've copied the `best.pt` file to the `YOLO-Interface` directory.

2. Navigate to the `YOLO-Interface` directory.

3. Run the application:
   ```
   python app.py
   ```

This will start the interface for using your trained YOLO model.
