import os
from utils._1_File_Mangement import * 
from utils._2_Extract_Grandstaff import * 
from utils._3_StaffnBar_Detection import *
from utils._4_Detect_Clef import *
from utils._5_Detect_Room_Limitor import *
from utils._6_Detect_Accidental import *
from utils._7_Detect_Beat import *
from utils._8_Pitch_Identification import *
from utils._9_Play_Order import *
from utils._10_Play_OnlyUpper import *

from utils._Ex_Detect_Crop_MusicElement import *

#file_input = "Twinkle_Twinkle_Little_Star"
file_input = ["Twinkle_Twinkle_Little_Star","Happy_birth_day","London_Bridge","If_You_Happy_And_You_Know_It","Baa_Baa_Black_Sheep","Old_MacDonald","Ode_To_Joy","Canon_In_D"]
file_input = file_input[0]
project_path = os.path.join("Sheet",file_input)
#Main: Twinkle (2) + London_Bridge (2) + Ode_To_Joy (4) + Canon_in_D (4)
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
    
    All_NOTE_DICT = []
    #Go through each Grand_Staff
    grand_staffs = [g for g in os.listdir(GRAND_STAFF_PATH) if g.lower().endswith("png")]
    for gs_num, grand_staff in enumerate(grand_staffs):
        print(grand_staff)
        #Staff Lines Detection --> Image and Position
        [upper_staff_boxes, lower_staff_boxes], staff_detected, cleaned_horizontal,staff_spacing = Staff_Position(GRAND_STAFF_PATH,grand_staff,gs_num,project_path)
        # Detect Bar Lines --> Image and Position
        only_element_img,barline_pos,music_element_path = Bar_Line_Detection(GRAND_STAFF_PATH, gs_num, staff_detected, cleaned_horizontal,project_path,9)

        gs_image_path = os.path.join(GRAND_STAFF_PATH, grand_staff)
        print(gs_image_path)
        gs_image = cv2.imread(gs_image_path, cv2.IMREAD_GRAYSCALE)
        print(gs_image.shape,only_element_img.shape)

        # Process and save detected elements
        object_pos_list,annotated_image = Music_Element_Detection(gs_image, gs_num, only_element_img, project_path)

        #Detect the Clef Key (update the object_pos_list)
        object_pos_list, Clef_Dictionary, clef_detected_image = Clef_Matching(object_pos_list, gs_image, project_path, file_input,gs_num)
        print(Clef_Dictionary)
        Pitch_Indicator_Dictionary = Staff_Pitch(Clef_Dictionary, upper_staff_boxes, lower_staff_boxes, gs_image)
        #print(Pitch_Indicator_Dictionary)

        #Detect Room Limitor
        object_pos_list, Room_Type, room_detected_image = Room_Matching(object_pos_list, clef_detected_image, project_path, file_input, gs_num)
        #print(Room_Type)

        #Detect Accidental
        object_pos_list, Accidental_Dict, accidental_detected_image = Accidental_Matching(object_pos_list, room_detected_image, project_path, file_input, gs_num)
        #print(Accidental_Dict)

        #Detect Beat
        object_pos_list, Beat_Dict, beat_detected_image = Beat_Matching(object_pos_list, accidental_detected_image, project_path, file_input, gs_num)
        #print(Beat_Dict)

        #Detect NoteHead
        NoteHead_Dictionary, notehead_detected_image = NoteHead_Matching(Beat_Dict, beat_detected_image, project_path, file_input, gs_num)
        #print(NoteHead_Dictionary)

        #Pitch_Detection
        Note_Dict = Pitch_Detection(NoteHead_Dictionary, gs_image, Pitch_Indicator_Dictionary, staff_spacing)
        Save_Pitch_Annotated(Note_Dict, gs_num, gs_image, project_path)

        All_NOTE_DICT.append(Note_Dict)

    #Music Playback
    Play_Music(All_NOTE_DICT, bpm=60)
    #Play_Upper_Staff_Notes(All_NOTE_DICT, gs_image.shape[0], bpm=60)

if __name__ == "__main__":
    main()