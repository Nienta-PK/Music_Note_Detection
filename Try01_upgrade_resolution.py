import cv2
import os
import numpy as np
from cv2 import dnn_superres

def upscale_image(image_path, scale_factor=3, model_path="Upscalling_Model/FSRCNN_x3.pb"):
    """
    Upscales an image using FSRCNN with OpenCV's dnn_superres module.
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found! Please download it.")
        return None

    # Load FSRCNN model
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", scale_factor)

    # Read input image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Ensure 3-channel BGR

    if image is None:
        print(f"Error: Image not found at {image_path}")
        return None

    # Ensure image is uint8
    image = image.astype(np.uint8)

    # Upscale the image
    try:
        upscaled = sr.upsample(image)
        upscaled = upscaled.astype(np.uint8)  # Ensure output is uint8
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return None
    
    return image,upscaled

def sharpen_image(image):
    """
    Applies unsharp masking to enhance edges and improve clarity.
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # Apply slight blur
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)  # Enhance edges
    return sharpened

def laplacian_sharpen(image): # Not Work
    """
    Sharpens an image using Laplacian filtering.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened = cv2.subtract(image, laplacian.astype(np.uint8))
    return sharpened

def high_pass_sharpen(image):
    """
    Enhances edges using high-pass filtering.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])  # High-pass filter kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def compare_resolution(original, upscaled):
    """
    Compare the resolution of the original and upscaled images.
    """
    # Load images
    #original = cv2.imread(original_path)
    #upscaled = cv2.imread(upscaled_path)
    
    if original is None or upscaled is None:
        print("Error loading images. Check file paths.")
        return
    
    # Get dimensions
    orig_h, orig_w = original.shape[:2]
    upscaled_h, upscaled_w = upscaled.shape[:2]
    
    print(f"Original Resolution: {orig_w}x{orig_h}")
    print(f"Upscaled Resolution: {upscaled_w}x{upscaled_h}")

def smooth_jigger(image,itr):
    # Apply Gaussian Blur to reduce jagged edges
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply morphological closing to smooth the lines
    kernel = np.ones((3, 3), np.uint8)
    smoothed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=itr)
    return smoothed

def remove_verti_hori_noise(image):
    # Define horizontal and vertical kernels
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Wide kernel for horizontal lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))    # Tall kernel for vertical lines

    # Apply morphological opening to detect horizontal and vertical noise
    horizontal_noise = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_noise = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Remove detected noise
    cleaned_image = cv2.subtract(image, horizontal_noise)
    cleaned_image = cv2.subtract(cleaned_image, vertical_noise)
    return cleaned_image

def main(img_path,factor,model):
    origin_img, upscaled_image = upscale_image(img_path, scale_factor=factor, model_path=f"Upscalling_Model/{model.upper()}_x{factor}.pb")
    if upscaled_image is not None:
        sharpened = high_pass_sharpen(upscaled_image)  # Use Unsharp Masking
        cv2.imwrite("sharpened_upscaled.png", sharpened)
        print("Sharpened image saved!")
    cv2.imwrite('upgrade_resolution.png',upscaled_image)
    compare_resolution(upscaled=upscaled_image,original=origin_img)

if __name__ == "__main__":
    main("dataset\Happy_birth_day.png",factor=3,model='fsrcnn')
    #compare_resolution("dataset/Happy_birth_day.png", "upgrade_resolution.png")
    #compare_resolution("Twinkle_Twinkle_Little_Star\page_1.png", "upgrade_resolution.png")