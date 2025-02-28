"""
1.Tkinter Dragdrop file into dataset (only pdf/png)
2.Select the file that user want in the combo_box
3.Check the dataset if it is png or pdf
    * if png then create note_project floder
    * if pdf then convert to png and save to note_project floder as page_number
4.Enhance Image Resolution
    
Input:
* pdf/png note sheet

Output:
* Project Folder
* Music Note Page
"""
import sys
sys.path.append(r'D:\2_2\Project\CV-music_note_extraction\Music_Note_Detection')    
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import Try01_upgrade_resolution as ur

def get_file(file_np,DATASET_FOLDER):
    files = os.listdir(DATASET_FOLDER)
    for file in files:
        file_name, suffix = file.split(".")
        if file_name == file_np:
            print(f"File Founded: {file_name}")
            return file,suffix
    print("File Doesn't Exist")
    return 0,0

def save_convert_file(file, suffix, DATASET_FOLDER, POPPLER_PATH,threshold_value = 180):
    note_path = os.path.join(DATASET_FOLDER, file)
    output_folder = os.path.splitext(file)[0]
    os.makedirs(output_folder, exist_ok=True)
    
    if suffix == "png":
        img = cv2.imread(f"dataset/{file}")

        if img is None:
            print("Error: Unable to load image.")
            return None
        
        output_path = os.path.join(output_folder, "page_1.png")
        
        # Get dimensions
        orig_h, orig_w = img.shape[:2]
        factor = 1  # Default factor

        if orig_w <= 750:
            factor = 4
        elif orig_w <= 1000:
            factor = 3
        elif orig_w <= 1500:
            factor = 2

        model = ['espcn', 'fsrcnn']

        # Upscale / Enhance Resolution
        if factor > 1:
            origin_img, upscaled_image = ur.upscale_image(
                f"dataset/{file}",
                scale_factor=factor,
                model_path=f"Upscalling_Model/{model[1].upper()}_x{factor}.pb"
            )
            if upscaled_image is not None:
                sharpened = ur.sharpen_image(upscaled_image)  # Use Unsharp Masking
                cv2.imwrite("sharpened_upscaled.png", sharpened)
                print("Sharpened image saved!")

                # Convert to grayscale before thresholding
                sharpened_gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
                _, thresholded_img = cv2.threshold(sharpened_gray, threshold_value, 255, cv2.THRESH_BINARY)
                
                # Smooth Jigger
                smoothed_jigger = ur.smooth_jigger(thresholded_img,itr=2)
                cv2.imwrite('upgrade_resolution.png', smoothed_jigger)
                ur.compare_resolution(upscaled=upscaled_image, original=origin_img)
                img = smoothed_jigger  # Use processed image

        # Save
        cv2.imwrite(output_path, img)
        return output_folder

    if suffix == "pdf":
        try:
            images = convert_from_path(note_path, dpi=300, poppler_path=POPPLER_PATH)
            
            for i, img in enumerate(images):
                img_cv = np.array(img)  # Convert PIL to NumPy
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                
                output_path = os.path.join(output_folder, f"page_{i+1}.png")
                
                # Get dimensions
                orig_h, orig_w = img_cv.shape[:2]
                factor = 1  # Default factor

                if orig_w <= 750:
                    factor = 4
                elif orig_w <= 1000:
                    factor = 3
                elif orig_w <= 1500:
                    factor = 2

                model = ['espcn', 'fsrcnn']

                # Upscale / Enhance Resolution
                if factor > 1:
                    origin_img, upscaled_image = ur.upscale_image(
                        f"dataset/{file}",
                        scale_factor=factor,
                        model_path=f"Upscalling_Model/{model[1].upper()}_x{factor}.pb"
                    )
                    if upscaled_image is not None:
                        sharpened = ur.sharpen_image(upscaled_image)  # Use Unsharp Masking
                        cv2.imwrite("sharpened_upscaled.png", sharpened)
                        print("Sharpened image saved!")

                        # Convert to grayscale before thresholding
                        sharpened_gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
                        _, thresholded_img = cv2.threshold(sharpened_gray, threshold_value, 255, cv2.THRESH_BINARY)

                        cv2.imwrite('upgrade_resolution.png', thresholded_img)
                        ur.compare_resolution(upscaled=upscaled_image, original=origin_img)
                        img_cv = thresholded_img  # Use processed image
                
                # Save the final processed image
                cv2.imwrite(output_path, img_cv)

            print("Success:", f"PDF converted successfully! Images saved in '{output_folder}'")
            return output_folder

        except Exception as e:
            print("Error:", f"Conversion failed: {e}")
            return None
        
def main(file_input,DATASET_FOLDER,POPPLER_PATH):
    file,suffix = get_file(file_input,DATASET_FOLDER)
    NOTE_PNG_PATH = save_convert_file(file,suffix,DATASET_FOLDER,POPPLER_PATH,threshold_value = 200)

if __name__ == "__main__":
    file_input = ["Twinkle_Twinkle_Little_Star","Happy_birth_day"]
    DATASET_FOLDER = "dataset"
    POPPLER_PATH = r"C:\tools\poppler\Library\bin"  # Adjust this path if necessary
    main(file_input[1],DATASET_FOLDER,POPPLER_PATH)