import os
from pdf2image import convert_from_path
from _1_File_Mangement import * 
from _2_Extract_Grandstaff import * 

file_input = "Twinkle_Twinkle_Little_Star"
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

    #Extract Music Element In Ordered

if __name__ == "__main__":
    main()