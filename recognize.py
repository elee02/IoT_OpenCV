import face_recognition
import pickle
import cv2

# Load the known encodings and names (use the code from the previous step)
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Load an image in which we want to recognize faces
test_image = face_recognition.load_image_file("./data/test_images/common.jpg")

# Find all face locations and their corresponding encodings in the test image
face_locations = face_recognition.face_locations(test_image, model="cnn")
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert the image to BGR for OpenCV display, because face_recognition uses RGB by default
test_image_cv2 = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

# Loop through all the faces found in the test image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    
    # Compare the detected face to the known encodings
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"  # Default name if no match is found

    # Use the first match found to identify the face
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a rectangle for the face and label it with the corresponding name
    cv2.rectangle(test_image_cv2, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(test_image_cv2, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow('Face Recognition', test_image_cv2)

# Wait for the press of the 'q' key
while True:
    key = cv2.waitKey(1) & 0xFF  # Wait for a short period and get the key pressed
    if key == ord('q'):  # If 'q' is pressed, break the loop
        break
cv2.destroyAllWindows()  # Close all OpenCV windows
