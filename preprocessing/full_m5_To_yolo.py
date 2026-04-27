import os
import shutil
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
############################################################################################

def convert_old(size, bbox):  # convert xyxy to xywh
        # size: (width， length)
        center_x, center_y = (bbox[0] + bbox[1]) / 2.0 - 1, (bbox[2] + bbox[3]) / 2.0 - 1
        center_w, center_h = bbox[1] - bbox[0], bbox[3] - bbox[2]
        return center_x / size[0], center_y / size[1], center_w / size[0], center_h / size[1]
    
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 
    y = (box[2] + box[3])/2.0 
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x,y,w,h #annotation format x==col values, y = row values

def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)



def get_class_and_bbox(xml_label_path,label_map, save_path):
    try:
        tree = ET.parse(xml_label_path)
    except:
        #print("failed to read: ", xml_label_path)
        return 0
    text_filename = str(save_path).replace("xml","txt")
    tree = ET.parse(xml_label_path)
    root = tree.getroot()

    data = []
    for size in root.findall("size"):
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        category_id = label_map[name.lower()]  # Use the name as the category ID
        box = [xmin,xmax,ymin,ymax]
        x, y, w, h = convert((width,height),box)
        data.append(f"{category_id} {x} {y} {w} {h}\n")
                
    # Save the data to a JSON file
    with open(text_filename, "w") as text_file:
        text_file.writelines(data)
            
    return 1
#############################################################################

def make_dirs(parent):
    #image dirs for china at each resolution and chucnk
    mkdir(parent/'China'/'images'/'test'/'100x')
    mkdir(parent/'China'/'images'/'test'/'400x')
    mkdir(parent/'China'/'images'/'test'/'1000x')

    mkdir(parent/'China'/'images'/'train'/'100x')
    mkdir(parent/'China'/'images'/'train'/'400x')
    mkdir(parent/'China'/'images'/'train'/'1000x')

    mkdir(parent/'China'/'images'/'val'/'100x')
    mkdir(parent/'China'/'images'/'val'/'400x')
    mkdir(parent/'China'/'images'/'val'/'1000x')

    #label dirs for china at each resolution and chucnk
    mkdir(parent/'China'/'labels'/'test'/'100x')
    mkdir(parent/'China'/'labels'/'test'/'400x')
    mkdir(parent/'China'/'labels'/'test'/'1000x')

    mkdir(parent/'China'/'labels'/'train'/'100x')
    mkdir(parent/'China'/'labels'/'train'/'400x')
    mkdir(parent/'China'/'labels'/'train'/'1000x')

    mkdir(parent/'China'/'labels'/'val'/'100x')
    mkdir(parent/'China'/'labels'/'val'/'400x')
    mkdir(parent/'China'/'labels'/'val'/'1000x')

    #image dirs for olympus at each resolution and chucnk
    mkdir(parent/'Olympus'/'images'/'test'/'100x')
    mkdir(parent/'Olympus'/'images'/'test'/'400x')
    mkdir(parent/'Olympus'/'images'/'test'/'1000x')

    mkdir(parent/'Olympus'/'images'/'train'/'100x')
    mkdir(parent/'Olympus'/'images'/'train'/'400x')
    mkdir(parent/'Olympus'/'images'/'train'/'1000x')

    mkdir(parent/'Olympus'/'images'/'val'/'100x')
    mkdir(parent/'Olympus'/'images'/'val'/'400x')
    mkdir(parent/'Olympus'/'images'/'val'/'1000x')
    
    #label dirs for olympus at each resolution and chucnk
    mkdir(parent/'Olympus'/'labels'/'test'/'100x')
    mkdir(parent/'Olympus'/'labels'/'test'/'400x')
    mkdir(parent/'Olympus'/'labels'/'test'/'1000x')

    mkdir(parent/'Olympus'/'labels'/'train'/'100x')
    mkdir(parent/'Olympus'/'labels'/'train'/'400x')
    mkdir(parent/'Olympus'/'labels'/'train'/'1000x')

    mkdir(parent/'Olympus'/'labels'/'val'/'100x')
    mkdir(parent/'Olympus'/'labels'/'val'/'400x')
    mkdir(parent/'Olympus'/'labels'/'val'/'1000x')
   
def preprocess_full_data( parent, output_root, label_map):
    
    # mapping from your folder names to script's expected names
    subdir_map = {
        'HCM': 'Olympus',
        'LCM': 'China'
    }
    
    make_dirs(output_root)
    #image_dir = parent / "Images"
    #image_dir = Path('/content/drive/MyDrive/M5_dataset/Images')
    # replace image_dir usage with a dict
    image_dirs = {
        'HCM': Path('/content/drive/MyDrive/M5_dataset/Images/HCM (1)'),  # the one with images
        'LCM': Path('/content/drive/MyDrive/M5_dataset/Images/LCM')
    }
    label_dir = parent / "Annotations"

    for dir in os.listdir(parent):
        if dir == "Annotations":
            for sub_dir in os.listdir(parent / dir):  # sub_dir = HCM or LCM
                
                out_sub = subdir_map[sub_dir]  # ← mapped name for output
                
                for _t_ in tqdm(os.listdir(parent / dir / sub_dir)):
                    image_list_all = []
                    for magnification in os.listdir(parent / dir / sub_dir / _t_):
                        if magnification != '1000x':  # ← add this
                            continue
                        count = 0
                        image_list = []
                        final_img_path = image_dirs[sub_dir] / _t_ / magnification
                        xml_label_path = label_dir / sub_dir / _t_ / magnification   # read with HCM/LCM
                        if not final_img_path.exists() or not xml_label_path.exists():
                            print(f"Skipping {sub_dir}/{_t_}/{magnification} — path missing")
                            continue
                        all_labels = sorted(os.listdir(xml_label_path))
                        all_imgs = sorted(os.listdir(final_img_path))

                        for img, l in zip(all_imgs, all_labels):
                            label = label_dir / sub_dir / _t_ / magnification / l
                            save_path = output_root / out_sub / 'labels' / _t_ / magnification / l  # ← out_sub
                            flag = get_class_and_bbox(label, label_map, save_path)
                            if flag:
                                image_path = image_dirs[sub_dir] / _t_ / magnification / img
                                img_copy_path = output_root / out_sub / 'images' / _t_ / magnification / img  # ← out_sub
                                if img_copy_path.exists():
                                    image_list.append(f'images/{_t_}/{magnification}/{img}\n')
                                    continue  # ← skips the copyfile below
                                shutil.copyfile(image_path, img_copy_path)
                                image_list.append(f'images/{_t_}/{magnification}/{img}\n')
                            else:
                                count += 1
                        image_list_all.extend(image_list)

                    with open(output_root / out_sub / f'yolo_{_t_}.txt', 'w') as f:  # ← out_sub
                        f.writelines(image_list_all)
                    print(count, " xml files could not be read in ", _t_)

                with open(output_root / out_sub / 'labels' / 'classes.txt', 'w') as f:  # ← out_sub
                    for k, v in label_map.items():
                        f.write(f'{k}\n')

if __name__ == '__main__':
    #root_dir = Path(__file__).parent
    """
    Instruction:
    1: Download the original cityscapes dataset like the following structure:
    -- cityscapes
       -- gtFine
       -- leftImg8bit
       -- vehicle_trainextra
       -- ...
    2: Change the root_dir to your dataset's path
    3: python cityscapes_to_yolo.py
    """
    
    label_map = {"gametocyte": 0, 'schizont': 1,'trophozoite': 2,'ring': 3}

    input_root = Path('/content/drive/MyDrive/MIC/Project/malaria_dataset/M5_dataset')          # 👈 original dataset
    output_root = Path('datasets/malaria_dataset')
    preprocess_full_data(input_root, output_root, label_map)