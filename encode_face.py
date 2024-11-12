import face_recognition
import os
import pickle

# Path to your images, where each person's images are inside their respective folder
image_directory = "./data/images"

# Dictionary to store encodings for each person (multiple images per person)
face_data = {
    "names": [],
    "encodings": []
}

# Loop through each folder (person)
for person_folder in os.listdir(image_directory):
    # Create a list to hold this person's multiple face encodings
    person_face_encodings = []
    
    # Folder path for the person
    person_folder_path = os.path.join(image_directory, person_folder)
    
    # Process each image (front, left, right) in the person's folder
    for image_name in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_name)

        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Get the face encodings (assuming there's exactly one face per image)
        face_encodings = face_recognition.face_encodings(image, model="cnn")
        
        if len(face_encodings) == 0:
            print(f"No faces found in the image: {image_name}. Skipping...")
            continue

        # For each image, get the encoding and save it
        face_encoding = face_encodings[0]
        person_face_encodings.append(face_encoding)
        print(f"Encoding Successful: {image_path}")
    
    # Add this person's multiple encodings to the overall list of known_encodings
    # We store all the encodings in one list, regardless of how many we have
    face_data["encodings"].extend(person_face_encodings)
    
    # Store the name as many times as we have face encodings for this person
    face_data["names"].extend([person_folder] * len(person_face_encodings))

# Save the encodings and names using pickle
with open("face_encodings_multi.pkl", "wb") as f:
    pickle.dump(face_data, f)

print("Successfully saved multiple face encodings for each person!")