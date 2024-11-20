import h5py
import numpy as np
import os
from PIL import Image
import json

CLASS2ID = {
    "cracker_box": 97.0,
    "sugar_box": 1.0,
    "tomato_soup_can": 2.0,
    "mustard_bottle": 3.0,
    "potted_meat_can": 4.0,
    "banana": 5.0,
    "bowl": 6.0,
    "mug": 7.0,
    "power_drill": 8.0,
    "strawberry": 11.0,
    "apple": 12.0,
    "lemon": 13.0,
    "peach": 14.0,
    "pear": 15.0,
    "orange": 16.0,
    "plum": 17.0,
    "knife": 18.0,
    "phillips_screwdriver": 19.0,
    "flat_screwdriver": 20.0,
    "racquetball": 21.0,
    "b_cups": 22.0,
    "d_cups": 23.0,
    "a_toy_airplane": 24.0,
    "c_toy_airplane": 25.0,
    "d_toy_airplane": 26.0,
    "f_toy_airplane": 27.0,
    "h_toy_airplane": 28.0,
    "i_toy_airplane": 29.0,
    "j_toy_airplane": 30.0,
    "k_toy_airplane": 31.0,
    "light_bulb": 32.0,
    "cables_in_transparent_bag": 37.0,
    "cables": 38.0,
    "wire_cutter": 39.0,
    "desinfection": 40.0,
    "hairspray": 41.0,
    "handcream": 42.0,
    "toothpaste": 43.0,
    "toydog": 44.0,
    "sponge": 45.0,
    "pneumatic_cylinder": 46.0,
    "airfilter": 47.0,
    "coffeefilter": 48.0,
    "wash_glove": 49.0,
    "wash_sponge": 50.0,
    "garbage_bags": 51.0,
    "deo": 52.0,
    "cat_milk": 53.0,
    "bottle_glass": 54.0,
    "shaving_cream": 56.0,
    "chewing_gum_with_spray": 57.0,
    "lighters": 58.0,
    "cream_soap": 59.0,
    "box_1": 60.0,
    "box_2": 61.0,
    "box_3": 62.0,
    "box_4": 63.0,
    "box_5": 64.0,
    "box_6": 65.0,
    "box_7": 66.0,
    "box_8": 67.0,
    "glass_cup": 68.0,
    "tennis_ball": 69.0,
    "wineglass": 71.0,
    "handsaw": 72.0,
    "lipcare": 73.0,
    "woodcube_a": 74.0,
    "lipstick": 75.0,
    "nosespray": 76.0,
    "tape": 77.0,
    "clamp": 79.0,
    "clamp_small": 83.0,
    "clamp_big": 84.0,
    "glasses": 85.0,
    "crayons": 86.0,
    "marker_big": 87.0,
    "marker_small": 88.0,
    "greek_busts": 89.0,
    "object_wrapped_in_foil": 90.0,
    "bubble_wrap": 92.0,
    "woodblock_a": 93.0,
    "woodblock_b": 94.0,
    "woodblock_c": 95.0,
    "mannequin": 96.0
}

ID2CLASS = {v: k for k, v in CLASS2ID.items()}

ID2COLOR = {
    1: (220 / 255, 20 / 255, 60 / 255),
    2: (119 / 255, 11 / 255, 32 / 255),
    3: (0.0, 0.0, 142 / 255),
    4: (0.0, 0.0, 230 / 255),
    5: (128 / 255, 64 / 255, 128 / 255),
    6: (0.0, 60 / 255, 100 / 255),
    7: (0.0, 80 / 255, 100 / 255),
    8: (0.0, 0.0, 70 / 255),
    11: (250 / 255, 170 / 255, 30 / 255),
    12: (250 / 255, 170 / 255, 160 / 255),
    13: (70 / 255, 70 / 255, 70 / 255),
    14: (153 / 255, 153 / 255, 153 / 255),
    15: (180 / 255, 165 / 255, 180 / 255),
    16: (150 / 255, 100 / 255, 100 / 255),
    17: (150 / 255, 120 / 255, 90 / 255),
    18: (250 / 255, 170 / 255, 160 / 255),
    19: (250 / 255, 170 / 255, 30 / 255),
    20: (220 / 255, 220 / 255, 0.0),
    21: (107 / 255, 142 / 255, 35 / 255),
    22: (152 / 255, 251 / 255, 152 / 255),
    23: (70 / 255, 130 / 255, 180 / 255),
    24: (220 / 255, 20 / 255, 60 / 255),
    25: (255 / 255, 0.0, 0.0),
    26: (0.0, 0.0, 142 / 255),
    27: (0.0, 0.0, 70 / 255),
    28: (0.0, 60 / 255, 100 / 255),
    29: (0.0, 0.0, 90 / 255),
    30: (0.0, 0.0, 110 / 255),
    31: (0.0, 80 / 255, 100 / 255),
    32: (0.0, 0.0, 230 / 255),
    37: (153 / 255, 153 / 255, 153 / 255),
    38: (250 / 255, 170 / 255, 30 / 255),
    39: (220 / 255, 220 / 255, 0.0),
    40: (107 / 255, 142 / 255, 35 / 255),
    41: (152 / 255, 251 / 255, 152 / 255),
    42: (70 / 255, 130 / 255, 180 / 255),
    43: (220 / 255, 20 / 255, 60 / 255),
    44: (255 / 255, 0.0, 0.0),
    45: (0.0, 0.0, 142 / 255),
    46: (0.0, 0.0, 70 / 255),
    47: (0.0, 60 / 255, 100 / 255),
    48: (0.0, 0.0, 90 / 255),
    49: (0.0, 0.0, 110 / 255),
    50: (0.0, 80 / 255, 100 / 255),
    51: (0.0, 0.0, 230 / 255),
    52: (119 / 255, 11 / 255, 32 / 255),
    53: (250 / 255, 170 / 255, 160 / 255),
    54: (230 / 255, 150 / 255, 140 / 255),
    56: (153 / 255, 153 / 255, 153 / 255),
    57: (250 / 255, 170 / 255, 30 / 255),
    58: (220 / 255, 220 / 255, 0.0),
    59: (107 / 255, 142 / 255, 35 / 255),
    60: (152 / 255, 251 / 255, 152 / 255),
    61: (70 / 255, 130 / 255, 180 / 255),
    62: (220 / 255, 20 / 255, 60 / 255),
    63: (255 / 255, 0.0, 0.0),
    64: (0.0, 0.0, 142 / 255),
    65: (0.0, 0.0, 70 / 255),
    66: (0.0, 0.0, 90 / 255),
    67: (0.0, 0.0, 110 / 255),
    68: (0.0, 80 / 255, 100 / 255),
    69: (0.0, 60 / 255, 100 / 255),
    71: (0.0, 0.0, 230 / 255),
    72: (250 / 255, 170 / 255, 160 / 255),
    73: (230 / 255, 150 / 255, 140 / 255),
    74: (190 / 255, 153 / 255, 153 / 255),
    75: (153 / 255, 153 / 255, 153 / 255),
    76: (250 / 255, 170 / 255, 30 / 255),
    77: (220 / 255, 220 / 255, 0.0),
    79: (152 / 255, 251 / 255, 152 / 255),
    83: (70 / 255, 130 / 255, 180 / 255),
    84: (220 / 255, 20 / 255, 60 / 255),
    85: (255 / 255, 0.0, 0.0),
    86: (0.0, 0.0, 142 / 255),
    87: (0.0, 0.0, 70 / 255),
    88: (0.0, 0.0, 90 / 255),
    89: (0.0, 0.0, 110 / 255),
    90: (0.0, 80 / 255, 100 / 255),
    92: (0.0, 60 / 255, 100 / 255),
    93: (0.0, 0.0, 230 / 255),
    94: (153 / 255, 153 / 255, 153 / 255),
    95: (250 / 255, 170 / 255, 160 / 255),
    96: (230 / 255, 150 / 255, 140 / 255),
    97: (190 / 255, 153 / 255, 153 / 255)
}


def print_structure(name, obj):
    print("Name is: ",name)
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset content: {obj[:]}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")
        for sub_name, sub_item in obj.items():
            if isinstance(sub_item, h5py.Dataset):
                print(f"  Sub-item name: {sub_name}")
                print(f" Sub-item is a dataset: {sub_item[:]}")
            elif isinstance(sub_item, h5py.Group):
                print(f"  Sub-group: {sub_name}")

def load_all_images(dir, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for ifl in range(0,17):
        print(f"IFL: {ifl}")
        path = os.path.join(dir, f"data_ifl_{ifl}", "mnt", "data1", "data_ifl_real")   
        for scene in os.listdir(path):
            scene_path = os.path.join(path, scene)
            print(f"Scene: {scene}")
            if os.path.isdir(scene_path):
                for file in os.listdir(scene_path):
                    if file.endswith("rgb.png"):
                        file_path = os.path.join(scene_path, file)
                        image = Image.open(file_path)
                        new_path = f"{scene}.png"
                        image.save(os.path.join(new_dir, new_path))
                        print(f"Image saved to {os.path.join(new_dir, scene)}")
                        break
      
def load_all_labels(dir):
    categories_set = set()
    for ifl in range(0,17):
        path = os.path.join(dir, f"data_ifl_{ifl}", "mnt", "data1", "data_ifl_real")   
        for scene in os.listdir(path):
            scene_path = os.path.join(path, scene)
            if os.path.isdir(scene_path):
                for file in os.listdir(scene_path):
                    if file.endswith("scene.hdf5"):
                        file_path = os.path.join(scene_path, file)
                        with h5py.File(file_path, 'r') as f:
                            objects_group = f['objects']
                            categoreis = objects_group['categories']
                            categories_set.update(categoreis)
                            break
    return categories_set
                 
def create_labels_file(dir, labels_file):
    labels= {}
    for ifl in range(0,17): 
        path = os.path.join(dir, f"data_ifl_{ifl}", "mnt", "data1", "data_ifl_real")   
        for scene in os.listdir(path):
            scene_path = os.path.join(path, scene)
            if os.path.isdir(scene_path):
                for file in os.listdir(scene_path):
                    if file.endswith("scene.hdf5"):
                        file_path = os.path.join(scene_path, file)
                        with h5py.File(file_path, 'r') as f:
                            objects_group = f['objects']
                            categoreis = objects_group['categories']
                            labels[scene] = categoreis[:].tolist()
                            break
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)

def get_scenes_with_single_category(labels_file):
    scene_for_each_category = {}

    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    for scene, categories in labels.items():
        if len(categories) ==  1:
            if categories[0] in scene_for_each_category:
                continue
            else:
                scene_for_each_category[categories[0]] = [scene]
        else:
            continue

    return scene_for_each_category
        
def create_query_set(labels_file, query_set_file, dir):
    if not os.path.exists(query_set_file):
        os.makedirs(query_set_file)
    scenes = get_scenes_with_single_category(labels_file)
    for category, scene in scenes.items():
        scene_path = os.path.join(dir, f"{scene[0]}.png")
        im = Image.open(scene_path)
        new_file_path = os.path.join(query_set_file, f"{category}_{ID2CLASS[category]}_1.png")
        im.save(new_file_path)

def create_test_images(source_dir, new_dir, query_scenes):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for scene in os.listdir(source_dir):
        scene_name = scene.split(".")[0]
        if scene_name not in query_scenes:
            scene_path = os.path.join(source_dir, scene)
            im = Image.open(scene_path)
            new_file_path = os.path.join(new_dir, scene)
            im.save(new_file_path)


if __name__ == '__main__':

    # Load all images from the dataset
    dir = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real"
    #load_all_images(dir, "MGN_images")

    '''
    # Create labels file
    #create_labels_file(dir, "MGN_labels.json")

   # create_query_set("MGN/MGN_labels.json", "Queries/MGN_query_set", "MGN/MGN_images")
    scenes = get_scenes_with_single_category("MGN/MGN_labels.json")
    scenes = [scene for sublist in scenes.values() for scene in sublist]
    create_test_images("MGN/MGN_images", "Test/MGN_test", scenes) 

    '''
    file_path = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real\\data_ifl_0\\mnt\\data1\\data_ifl_real\\scene1\\3_scene.hdf5"
    #file_path = "C:\\Users\\cm03009\\Downloads\\MetaGraspNet_Real (1)\\MetaGraspNet_Real\\data_ifl_real_0_588.zip\\scene578\\"
    file_path2 = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real\\data_ifl_0\\mnt\\data1\\data_ifl_real\\scene0\\34.npz"
    file_path2 = "C:\\Users\\cm03009\\Downloads\\28.npz"

    file_path = "C:\\Users\\cm03009\\Downloads\\data_ifl_16\\mnt\\data1\\data_ifl_real\\scene800\\17.npz"
    with np.load(file_path) as data:
        depth = data['depth']
        print(f"Depth: {depth}")
        try:
            instances_objects = data['instances_objects']
            print(f"Instances objects: {instances_objects}")
        except:
            print(f"WARNING : instances_objects not in file")
            instances_objects = np.zeros_like(depth)
        try:
            instances_semantic = data['instances_semantic']
            print(f"Instances semantic: {instances_semantic}")
        except:
            print(f"WARNING : instances_semantic not in file")
            instances_semantic = np.zeros_like(depth)

    '''
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_structure)

    def read_npz(file_path):
        npz = np.load(file_path)
        print(f"Contents of {file_path}:")
        for key in npz.files:
            print(f"{key}:")
            print(f"  Data type: {npz[key].dtype}")
            print(f"  Data shape: {npz[key].shape}")
            print(npz[key])
        print("\n")

    read_npz(file_path2)
    '''
    
