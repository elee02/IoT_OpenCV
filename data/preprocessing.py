import os
import cv2

# Directory containing the images
INPUT_DIRECTORY = './data/images/'  # Change as needed

# Define the path for the preprocessed images
OUTPUT_DIRECTORY = './data/preprocessed_images/'  # Save preprocessed images here

# Desired output image size
TARGET_SIZE = (640, 480)  # Typical size that's effective for face_recognition

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

def preprocess_image(input_image_path, output_image_path, target_size):
    """
    Preprocess an image by reading it, resizing it to target size, 
    and saving it to the output directory.
    """
    # Read the image using OpenCV
    image = cv2.imread(input_image_path)
    
    if image is None:
        print(f"Error: Unable to read the image {input_image_path}")
        return
    
    # Convert the image to RGB (face_recognition works with RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the TARGET_SIZE while maintaining the aspect ratio
    resized_image = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert back to BGR, as OpenCV uses BGR format for writing images
    resized_image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    
    # Save the preprocessed image in the BGR format
    cv2.imwrite(output_image_path, resized_image_bgr)
    print(f"Processed and saved: {output_image_path}")


def preprocess_images(input_directory, output_directory, target_size):
    """
    Walk through the input directory, preprocess each image,
    and store the preprocessed image in the output directory.
    """    
    # Traverse through input directory structure
    for person_name in os.listdir(input_directory):
        person_path = os.path.join(input_directory, person_name)

        if os.path.isdir(person_path):
            # Create output directory for this person
            person_output_directory = os.path.join(output_directory, person_name)
            if not os.path.exists(person_output_directory):
                os.makedirs(person_output_directory)

            # Traverse right/front/left images for this person
            for pose_image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, pose_image_name)

                # Define output image path
                output_image_path = os.path.join(person_output_directory, pose_image_name)

                # Preprocess the image
                preprocess_image(image_path, output_image_path, target_size)


if __name__ == "__main__":
    # Run the preprocessing for every image
    preprocess_images(INPUT_DIRECTORY, OUTPUT_DIRECTORY, TARGET_SIZE)
