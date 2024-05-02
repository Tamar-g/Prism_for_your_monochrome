from PIL import Image
import os


def convert_to_grayscale(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, dirs, files in os.walk(input_path):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg")):
                try:
                    img = Image.open(os.path.join(root, filename))
                    gray_img = img.convert('L')  # Convert to grayscale
                    relative_path = os.path.relpath(root, input_path)
                    output_folder = os.path.join(output_path, relative_path)

                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    output_file = os.path.join(output_folder, filename)
                    gray_img.save(output_file)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


# Specify the root input and output directories
input_root = fr'C:\Users\Final_Project\Desktop\color_archive'
output_root = fr'C:\Users\Final_Project\Desktop\black_and_white_arcive'

# Call the function
convert_to_grayscale(input_root, output_root)
