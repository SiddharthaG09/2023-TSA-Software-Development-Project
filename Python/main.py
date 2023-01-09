import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define the labels
labels = {0: "Happy", 1: "Sad"}

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process the frame for the model
    frame = cv2.resize(frame, (48, 48))
    frame = np.expand_dims(frame, axis=0)

    # Predict the emotion
    prediction = model.predict(frame)[0]
    label = labels[np.argmax(prediction)]

    # Display the frame with the label
    cv2.putText(frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    # Break the loop if the user presses "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
