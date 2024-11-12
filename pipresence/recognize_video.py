import face_recognition
import cv2
import pickle

# Load the known face encodings
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Open the video file
video_path = "/home/el02/IoT_OpenCV/data/test_videos/idris.mp4"
video_capture = cv2.VideoCapture(video_path)

# Loop through each frame from the video file
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Break the loop when no more frames are available
    if not ret:
        break

    # Convert the frame from BGR (OpenCV default) to RGB (face_recognition expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over each face found in the frame to see if it matches known faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the first match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face and label it with the person's name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Stop the video if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the video file
video_capture.release()
cv2.destroyAllWindows()
