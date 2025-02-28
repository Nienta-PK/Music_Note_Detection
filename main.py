import os
from pdf2image import convert_from_path
from utils._1_File_Mangement import * 
from utils._2_Extract_Grandstaff import * 
from utils._3_StaffnBar_Detection import *

#file_input = "Twinkle_Twinkle_Little_Star"
file_input = "Happy_birth_day"

DATASET_FOLDER = "dataset"
POPPLER_PATH = r"C:\tools\poppler\Library\bin"  # Adjust this path if necessary

def main():
    #Note Project_Selection
    file,suffix = get_file(file_input,DATASET_FOLDER)
    NOTE_PNG_PATH = save_convert_file(file,suffix,DATASET_FOLDER,POPPLER_PATH)
    print(NOTE_PNG_PATH)

    #Extract the Grand_Staff
    all_grand_staff = []
    pages = [p for p in os.listdir(NOTE_PNG_PATH) if p.lower().endswith('.png') and p[0:4]=="page"]
    print(pages)
    for i, page in enumerate(pages):
        detected_line_img = Line_Detection(NOTE_PNG_PATH, page, i)
        if detected_line_img is None:
            continue
        grandstaffs_onePage_list = Grand_Staff_Extraction(NOTE_PNG_PATH, page, detected_line_img, i)
        all_grand_staff.extend(grandstaffs_onePage_list)
    GRAND_STAFF_PATH = Save_Grand_Staffs(NOTE_PNG_PATH, all_grand_staff)
    print(GRAND_STAFF_PATH)

    #Go through each Grand_Staff
    grand_staffs = [g for g in os.listdir(GRAND_STAFF_PATH) if g.lower().endswith("png")]
    for gs_num, grand_staff in enumerate(grand_staffs):
        #Staff Lines Detection --> Image and Position
        [upper_staff_boxes, lower_staff_boxes], staff_detected, cleaned_horizontal,staff_spacing = Staff_Position(GRAND_STAFF_PATH,grand_staff,gs_num,file_input)
        # Detect Bar Lines --> Image and Position
        bar_lines,barline_pos = Bar_Line_Detection(GRAND_STAFF_PATH, gs_num, staff_detected, cleaned_horizontal)
    

if __name__ == "__main__":
    main()