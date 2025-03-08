"""
1.Go into grand_staff_number floder and see through each grand_staff
2.Staff_Thickness
3.Get Staff_Position & Staff_Spacing using morphologyEx
4.Remove the staff line
5.Process to get Barline & the music note stem if possible
6.

Input:
* Grand Staff Image

Output:
* Staff line thickness
* Staff / Barline Position
* Spacing
* detected staff / detected barlines image
* Image without staff
* Image without Staff & Barline
"""
import os 
import cv2
import numpy as np
import sys
sys.path.append(r'D:\2_2\Project\CV-music_note_extraction\Music_Note_Detection')    
import utils._0_upgrade_resolution as ur

def Staff_Thickness(image):
    """
    Detects the average thickness of staff lines.
    """
    # Create a horizontal kernel (thin)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    
    # Use morphological operations to extract staff lines
    staff_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Find contours of the staff lines
    contours, _ = cv2.findContours(staff_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thicknesses = [cv2.boundingRect(cnt)[3] for cnt in contours]  # Extract heights (thickness)
    
    # Compute average thickness
    return int(np.mean(thicknesses)) if thicknesses else 1  # Ensure at least 1px thickness

def Staff_Position(gs_folder, gs, gs_num, file_input):
    image_path = os.path.join(gs_folder, gs)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return [], [], None, None

    # Step 1: Binary thresholding
    _, binary = cv2.threshold(image, 165, 255, cv2.THRESH_BINARY_INV)
        
    # Step 2: Detect horizontal lines (staff lines)
    kernel_width = max(image.shape[1] // 5, 1)  # Ensure kernel width is at least 1 pixel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    print(f"Grandstaff {gs_num+1} has extracted the lines successfully")

    # Step 3: Find contours for detected staff lines
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return [], [], None, None
    
    # Step 4: Get bounding boxes for each detected line
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Step 5: Sort bounding boxes by Y-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[1])

    # Step 6: Compute gaps between lines
    gaps = [bounding_boxes[i+1][1] - (bounding_boxes[i][1] + bounding_boxes[i][3]) 
            for i in range(len(bounding_boxes) - 1)]
    
    # Step 7: Identify the split point between Upper and Lower staff
    if len(bounding_boxes) >= 10:  # Ensure there are enough staff lines
        split_index = 5  # Each staff should have 5 lines
    elif gaps:
        split_index = gaps.index(max(gaps)) + 1  # Fallback: Use max gap
    else:
        split_index = len(bounding_boxes) // 2  # Default fallback

    # Step 8: Compute staff spacing (median of detected gaps)
    staff_spacing = int(np.median(gaps)) if gaps else 1  # Ensure at least 1px spacing

    # Step 9: Separate bounding boxes into upper and lower staves
    upper_staff_boxes = bounding_boxes[:split_index]
    lower_staff_boxes = bounding_boxes[split_index:]

    # Step 10: Annotate the image with detected staff lines
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for box in upper_staff_boxes:
        x, y, w, h = box
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for upper staff

    for box in lower_staff_boxes:
        x, y, w, h = box
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for lower staff

    # Step 11: Save the annotated image
    annotated_path = os.path.join(gs_folder, "Annotated_Grandstaff")
    os.makedirs(annotated_path, exist_ok=True)
    annotated_save_path = os.path.join(annotated_path, f"Annotated_Grandstaff_{gs_num+1}.png")
    cv2.imwrite(annotated_save_path, annotated_image)
    print(f"Annotated image saved to {annotated_save_path}")

    # Step 12: Remove Staff Lines by subtracting detected lines (with dilation to clean artifacts)
    dilated_lines = cv2.dilate(detected_lines, np.ones((3, 3), np.uint8), iterations=1)
    cleaned_horizontal = cv2.subtract(binary, dilated_lines)

    # Step 13: Save the cleaned image without staff lines
    no_staff_path = os.path.join(file_input, "No_Staff")
    os.makedirs(no_staff_path, exist_ok=True)
    no_staff_save_path = os.path.join(no_staff_path, f"No_Staff_{gs_num+1}.png")
    cv2.imwrite(no_staff_save_path, cleaned_horizontal)
    print(f"Cleaned image saved to {no_staff_save_path}")

    # Step 14: Return staff positions, detected lines, cleaned image, and spacing
    staff_pos = [upper_staff_boxes, lower_staff_boxes]
    return staff_pos, detected_lines, cleaned_horizontal, staff_spacing

def Bar_Line_Detection(gs_path, gs_num, detected_line, bw_grandstaff,file_input, kernel_height):
    """
    Detects vertical barlines and replaces them with clean lines.
    """

    # Step 1: Staff_Thickness
    staff_thickness = Staff_Thickness(bw_grandstaff)

    # Step 2: Subtract Horizontal Lines (Keeps Vertical Structures)
    cleaned_horizontal = cv2.subtract(bw_grandstaff, detected_line)

    # Step 3: Smooth remaining noise
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    smoothed_image = cv2.morphologyEx(cleaned_horizontal, cv2.MORPH_CLOSE, closing_kernel)

    # Step 4: Detect Bar Lines
    vertical_kernel_height = staff_thickness * 10  # Adaptive kernel to capture barlines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))
    detected_vertical_lines = cv2.morphologyEx(smoothed_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Step 5: Find contours and filter short vertical lines
    contours, _ = cv2.findContours(detected_vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_vertical_lines = np.zeros_like(detected_vertical_lines)
    vertical_positions = []  # Store bar line positions
    
    # Keep only tall vertical elements (barlines)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= smoothed_image.shape[0] * 1/2:  # Threshold to remove short vertical lines (like stems)
            vertical_positions.append((x, y, w, h))  # Store positions
            cv2.drawContours(filtered_vertical_lines, [cnt], -1, 255, thickness=cv2.FILLED)
    
    # Step 6: Create clean vertical bar lines from top to bottom
    for x, _, w, _ in vertical_positions:
        cv2.line(filtered_vertical_lines, (x+w//2, 0), (x+w//2, smoothed_image.shape[0]), 255, thickness=w)
    
    # Step 7: Subtract bar lines from cleaned horizontal
    final_image = cv2.subtract(smoothed_image, filtered_vertical_lines)
    
    # Define erosion kernel
    erosion_kernel = np.ones((2, 2), np.uint8)  # Small kernel to erode noise
    final_image = cv2.erode(final_image, erosion_kernel, iterations=1)
    dilation_kernel = np.ones((2, 10), np.uint8)  # Small kernel to erode noise
    final_image = cv2.dilate(final_image, dilation_kernel, iterations=1)
    #close_kernel = np.ones((3, 2), np.uint8)
    #final_image = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    #final_image = ur.remove_verti_hori_noise(final_image)
    
    # Step 8: Apply Morphological Opening to Reduce Noise
    #noise_reduction_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    #final_image = cv2.morphologyEx(final_image, cv2.MORPH_OPEN, noise_reduction_kernel, iterations=1)
    
    # Step 9: Save Vertical Lines Detection
    vertical_path = os.path.join(gs_path, "Vertical_Lines")
    os.makedirs(vertical_path, exist_ok=True)
    save_path = os.path.join(vertical_path, f"Vertical_Lines_{gs_num+1}.png")
    cv2.imwrite(save_path, filtered_vertical_lines)
    print(f"Detected vertical lines saved to {save_path}")
    
    # Step 10: Save Final Image with Subtracted Bar Lines
    element_path = os.path.join(file_input, "Only_Music_Elements")
    os.makedirs(element_path, exist_ok=True)
    final_save_path = os.path.join(element_path, f"Final_Cleaned_{gs_num+1}.png")
    cv2.imwrite(final_save_path, final_image)
    print(f"Final cleaned image saved to {final_save_path}")
    
    return final_image,vertical_positions,element_path

def main(file_input, gs_path):
    grand_staffs = [g for g in os.listdir(gs_path) if g.lower().endswith("png")]
    for gs_num, grand_staff in enumerate(grand_staffs):
        [upper_staff_boxes, lower_staff_boxes], staff_detected, cleaned_horizontal,staff_spacing = Staff_Position(gs_path,grand_staff,gs_num,file_input)
        print(f"Upper_Staff_{gs_num+1}",upper_staff_boxes) #x, y, w, h
        print(f"LowerStaff_{gs_num+1}",lower_staff_boxes) #x, y, w, h
        print(f"Staff Spacing: {staff_spacing}")

        # Detect Bar Lines (Barlines & Note Stems)
        bar_lines,barline_pos,music_element_path = Bar_Line_Detection(gs_path, gs_num, staff_detected, cleaned_horizontal,file_input,9)
        print("************************************************")
        
if __name__ == "__main__":
    GRAND_STAFF_PATH = ["Twinkle_Twinkle_Little_Star\Grand_Staff_IMG","Happy_birth_day\Grand_Staff_IMG","Sheet\If_You_Happy_And_You_Know_It\Grand_Staff_IMG"]
    file_input = ["Twinkle_Twinkle_Little_Star","Happy_birth_day","If_You_Happy_And_You_Know_It"]
    x = 2
    main(file_input[x], GRAND_STAFF_PATH[x])
