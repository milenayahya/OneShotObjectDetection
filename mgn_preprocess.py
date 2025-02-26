import h5py
import numpy as np
import os
from PIL import Image
import json
from utils import convert_from_x1y1x2y2_to_coco, convert_masks_to_boxes
from matplotlib import pyplot as plt
from config import query_dir
import random

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
    "mannequin": 96.0,
    "condensator": 98.0,
    "allen_key": 99.0
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
    97: (190 / 255, 153 / 255, 153 / 255),
    98: (190 / 255, 153 / 255, 153 / 255),
    99: (0 , 0, 255 / 255)
}


############################################################################################
# Mapping to supercategories in a postprocessing step
ID2CLASS_post = {
    0: "food",
    1: "kitchenware",
    2: "tool",
    3: "model",
    4: "personal_care",
    5: "storage",
    6: "household_item"

}

ID2COLOR_post = {
    0: (1, 0, 0),       # Red for food
    1: (0, 1, 0),       # Green for kitchenware
    2: (0, 0, 1),       # Blue for tool
    3: (1, 1, 0),     # Yellow for model
    4: (1, 0, 1),     # Magenta for personal care
    5: (0, 1, 1),     # Cyan for storage
    6: (128/255, 0, 128/255)      # Purple for household item
}


CLASS2ID_post = {v: k for k, v in ID2CLASS_post.items()}

CAT_TO_SUPERCAT_post = {
    "cracker_box": "food",
    "sugar_box": "food",
    "tomato_soup_can": "food",
    "mustard_bottle": "food",
    "potted_meat_can": "food",
    "banana": "food",
    "strawberry": "food",
    "apple": "food",
    "lemon": "food",
    "peach": "food",
    "pear": "food",
    "orange": "food",
    "plum": "food",
    "cat_milk": "food",
    "bowl": "kitchenware",
    "mug": "kitchenware",
    "glass_cup": "kitchenware",
    "wineglass": "kitchenware",
    "coffeefilter": "kitchenware",
    "wash_glove": "kitchenware",
    "wash_sponge": "kitchenware",
    "power_drill": "tool",
    "knife": "tool",
    "phillips_screwdriver": "tool",
    "flat_screwdriver": "tool",
    "wire_cutter": "tool",
    "pneumatic_cylinder": "tool",
    "handsaw": "tool",
    "clamp": "tool",
    "clamp_small": "tool",
    "clamp_big": "tool",
    "racquetball": "model",
    "a_toy_airplane": "model",
    "c_toy_airplane": "model",
    "d_toy_airplane": "model",
    "f_toy_airplane": "model",
    "h_toy_airplane": "model",
    "i_toy_airplane": "model",
    "j_toy_airplane": "model",
    "k_toy_airplane": "model",
    "toydog": "model",
    "greek_busts": "model",
    "mannequin": "model",
    "desinfection": "personal_care",
    "hairspray": "personal_care",
    "handcream": "personal_care",
    "toothpaste": "personal_care",
    "shaving_cream": "personal_care",
    "cream_soap": "personal_care",
    "lipcare": "personal_care",
    "lipstick": "personal_care",
    "nosespray": "personal_care",
    "deo": "personal_care",
    "box_1": "storage",
    "box_2": "storage",
    "box_3": "storage",
    "box_4": "storage",
    "box_5": "storage",
    "box_6": "storage",
    "box_7": "storage",
    "box_8": "storage",
    "bubble_wrap": "storage",
    "object_wrapped_in_foil": "storage",
    "garbage_bags": "storage",
    "cables_in_transparent_bag": "storage",
    "light_bulb": "household_item",
    "cables": "household_item",
    "sponge": "household_item",
    "airfilter": "household_item",
    "lighters": "household_item",
    "tape": "household_item",
    "crayons": "household_item",
    "marker_big": "household_item",
    "marker_small": "household_item",
    "glasses": "household_item",
    "woodcube_a": "household_item",
    "woodblock_a": "household_item",
    "woodblock_b": "household_item",
    "woodblock_c": "household_item"
}

############################################################################################

CLASS2ID_pre = {
    "cracker_box": 97.0,
    "sugar_box": 1.0,
    "tomato_soup_can": 2.0,
    "mustard_bottle": 3.0,
    "potted_meat_can": 4.0,
    "banana": 5.0,
    "round_fruit": 6.0,
    "toy_airplane": 7.0,
    "power_drill": 8.0,
    "thin_tool": 9.0,
    "product": 10.0,
    "small_object": 11.0,
    "tableware": 12.0,
    "glass": 13.0,
    "white_box": 14.0,
    "pear": 15.0,
    "dark_box": 16.0,
    "clamp": 17.0,
    "wrap": 18.0,
    "marker": 19.0,
    "spray": 20.0,
    "racquetball": 21.0,
    "b_cups": 22.0,
    "d_cups": 23.0,
    "f_toy_airplane": 27.0,
    "h_toy_airplane": 28.0,
    "k_toy_airplane": 31.0,
    "cables_in_transparent_bag": 37.0,
    "cables": 38.0,
    "wire_cutter": 39.0,
    "handcream": 42.0,
    "toothpaste": 43.0,
    "toydog": 44.0,
    "sponge": 45.0,
    "pneumatic_cylinder": 46.0,
    "airfilter": 47.0,
    "wash_glove": 49.0,
    "wash_sponge": 50.0,
    "garbage_bags": 51.0,
    "deo": 52.0,
    "shaving_cream": 56.0,
    "chewing_gum_with_spray": 57.0,
    "lighters": 58.0,
    "cream_soap": 59.0,
    "tennis_ball": 69.0,
    "handsaw": 72.0,
    "lipcare": 73.0,
    "woodcube_a": 74.0,
    "nosespray": 76.0,
    "tape": 77.0,
    "glasses": 85.0,
    "crayons": 86.0,
    "greek_busts": 89.0,
    "woodblock_a": 93.0,
    "woodblock_b": 94.0,
    "woodblock_c": 95.0,
    "mannequin": 96.0,
    "condensator": 98.0
}
ID2CLASS_pre = {v: k for k, v in CLASS2ID_pre.items()}
CAT_TO_SUPERCAT_pre = {
    "a_toy_airplane": "toy_airplane",
    "c_toy_airplane": "toy_airplane",
    "apple": "round_fruit",
    "lemon": "round_fruit",
    "peach": "round_fruit",
    "orange": "round_fruit",
    "plum": "round_fruit",
    "strawberry": "round_fruit",
    "bowl": "tableware",
    "mug": "tableware",
    "knife": "thin_tool",
    "phillips_screwdriver": "thin_tool",
    "flat_screwdriver": "thin_tool",
    "h_toy_airplane": "small_object",
    "d_toy_airplane": "small_object",
    "i_toy_airplane": "small_object",
    "j_toy_airplane": "small_object",
    "glass_cup": "glass",
    "wineglass": "glass",
    "light_bulb": "glass",
    "bottle_glass": "glass",
    "box_1": "white_box",
    "box_2": "white_box",
    "box_3": "white_box",
    "box_4": "white_box",
    "box_5": "white_box",
    "box_8": "white_box",
    "box_6": "dark_box",
    "box_7": "dark_box",
    "desinfection": "spray",
    "hairspray": "spray",
    "marker_big": "marker",
    "marker_small": "marker",
    "lipstick": "marker",
    "cat_milk": "product",
    "coffeefilter": "product",
    "clamp": "clamp",
    "clamp_small": "clamp",
    "clamp_big": "clamp",
    "object_wrapped_in_foil": "wrap",
    "bubble_wrap": "wrap"
}

ID2COLOR_pre = {
    97.0: [1.0, 0.0, 0.0],    # cracker_box: Red
    1.0: [0.0, 1.0, 0.0],     # sugar_box: Green
    2.0: [0.0, 0.0, 1.0],     # tomato_soup_can: Blue
    3.0: [1.0, 1.0, 0.0],     # mustard_bottle: Yellow
    4.0: [1.0, 0.5, 0.0],     # potted_meat_can: Orange
    5.0: [0.5, 1.0, 0.5],     # banana: Light Green
    6.0: [0.5, 0.5, 1.0],     # round_fruit: Light Blue
    7.0: [1.0, 0.0, 1.0],     # toy_airplane: Magenta
    8.0: [0.5, 0.0, 0.5],     # power_drill: Purple
    9.0: [0.0, 1.0, 1.0],     # thin_tool: Cyan
    10.0: [0.6, 0.3, 0.2],    # product: Brown
    11.0: [0.8, 0.6, 0.4],    # small_object: Tan
    12.0: [1.0, 0.8, 0.8],    # tableware: Pink
    13.0: [0.7, 0.7, 0.7],    # glass: Gray
    14.0: [0.7, 0.7, 0.7],    # white_box: White
    15.0: [0.0, 0.0, 0.0],    # pear: Black
    16.0: [0.4, 0.2, 0.0],    # dark_box: Dark Brown
    17.0: [0.3, 0.3, 0.7],    # clamp: Slate Blue
    18.0: [0.7, 0.4, 0.4],    # wrap: Dusky Pink
    19.0: [0.5, 0.2, 0.8],    # marker: Violet
    20.0: [0.4, 0.7, 0.4],    # spray: Moss Green
    21.0: [0.9, 0.9, 0.3],    # racquetball: Golden Yellow
    22.0: [0.2, 0.6, 0.8],    # b_cups: Aquamarine
    23.0: [0.8, 0.2, 0.6],    # d_cups: Plum
    27.0: [0.3, 0.8, 0.5],    # f_toy_airplane: Mint Green
    28.0: [0.6, 0.3, 0.8],    # h_toy_airplane: Lavender
    31.0: [0.9, 0.6, 0.2],    # k_toy_airplane: Amber
    37.0: [0.5, 0.5, 0.5],    # cables_in_transparent_bag: Medium Gray
    38.0: [0.8, 0.5, 0.1],    # cables: Burnt Orange
    39.0: [0.3, 0.3, 0.3],    # wire_cutter: Charcoal Gray
    42.0: [1.0, 0.6, 0.6],    # handcream: Light Coral
    43.0: [0.2, 0.8, 1.0],    # toothpaste: Sky Blue
    44.0: [0.8, 0.4, 0.2],    # toydog: Terracotta
    45.0: [0.4, 0.6, 0.8],    # sponge: Steel Blue
    46.0: [0.6, 0.6, 1.0],    # pneumatic_cylinder: Light Periwinkle
    47.0: [0.6, 0.8, 0.6],    # airfilter: Pastel Green
    49.0: [1.0, 0.8, 0.4],    # wash_glove: Light Gold
    50.0: [0.8, 0.8, 0.2],    # wash_sponge: Mustard Yellow
    51.0: [0.5, 0.5, 0.0],    # garbage_bags: Olive Green
    52.0: [0.2, 0.5, 0.2],    # deo: Forest Green
    56.0: [0.9, 0.7, 0.4],    # shaving_cream: Apricot
    57.0: [0.6, 0.9, 0.6],    # chewing_gum_with_spray: Light Mint
    58.0: [1.0, 0.5, 0.0],    # lighters: Bright Orange
    59.0: [0.7, 0.9, 0.7],    # cream_soap: Mint
    69.0: [0.9, 0.9, 0.9],    # tennis_ball: Very Light Gray
    72.0: [0.3, 0.4, 0.7],    # handsaw: Indigo
    73.0: [0.9, 0.7, 0.8],    # lipcare: Pale Pink
    74.0: [0.8, 0.7, 0.5],    # woodcube_a: Beige
    76.0: [0.3, 0.6, 0.3],    # nosespray: Medium Green
    77.0: [0.7, 0.3, 0.3],    # tape: Rust Red
    85.0: [0.9, 0.9, 0.6],    # glasses: Pale Yellow
    86.0: [0.6, 0.3, 0.6],    # crayons: Mauve
    89.0: [0.8, 0.8, 1.0],    # greek_busts: Pale Lavender
    93.0: [0.5, 0.4, 0.3],    # woodblock_a: Wood Brown
    94.0: [0.6, 0.5, 0.4],    # woodblock_b: Light Brown
    95.0: [0.7, 0.6, 0.5],    # woodblock_c: Tan
    96.0: [0.8, 0.8, 0.9],    # mannequin: Soft Gray
    98.0: [0.5, 0.4, 0.3],
    99.0: [0.7, 0.3, 0.3]
}

############################################################################################

def get_viewpoints(dir):
    viewpoints = {}   
    for file in os.listdir(dir):
        scene = file.split("_")[0]
        viewpoint = file.split("_")[1].split(".")[0]
        viewpoints[scene] = viewpoint
    return viewpoints

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
                        viewpoint = file.split("_")[0]
                        image = Image.open(file_path)
                        new_path = f"{scene}_{viewpoint}.png"
                        image.save(os.path.join(new_dir, new_path))
                        print(f"Image saved to {os.path.join(new_dir,new_path)}")
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

def crop_object(image_id, bbox, dir, cat, i, new_dir):

    for file in os.listdir(dir):
        if file.startswith(f"scene{image_id}_"):
            scene_path = os.path.join(dir, file)
            im = Image.open(scene_path)
            x1, y1, w, h = bbox
            im = im.crop((x1, y1, x1+w, y1+h))
            new_file_name = os.path.join(new_dir, f"{CLASS2ID[cat]}_{cat}_{i+1}.png")
            im.save(new_file_name)
            break

def mgn_create_queries(labels_file, image_dir, min_size, num_objects_per_class):
    
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    categories = {k for k in CLASS2ID.keys()}

    dir = os.path.join(query_dir, f"MGN_{num_objects_per_class}_shot")
    if not os.path.exists(dir):
        os.makedirs(dir)

    for cat in categories:
        i = 0
        for ann in labels["annotations"]:
            if ann["category_id"] == CLASS2ID[cat]:
                bbox = ann["bbox"]
                if bbox[2] * bbox[3] >= min_size:
                    image_id = ann["image_id"]
                    crop_object(image_id, bbox, image_dir, cat, i, dir)
                    i += 1
                    if i == num_objects_per_class:
                        break                   


def get_test_scenes(dir):
    test_scenes= []
    for file in os.listdir(dir):
        scene = file.split(".")[0].split("_")[0]
        test_scenes.append(scene) 
    return test_scenes
             
def create_labels_file(dir, labels_file, test=False):
    labels = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    viewpoints = get_viewpoints("MGN/MGN_images")
    ann_id = 0
    if test:
        test_scenes = get_test_scenes("Test/MGN/MGN_test_images")
    for ifl in range(0,17): 
        path = os.path.join(dir, f"data_ifl_{ifl}", "mnt", "data1", "data_ifl_real")   
        for scene in os.listdir(path):
            if test and scene not in test_scenes:
                continue
            scene_path = os.path.join(path, scene)
            scene_nb  = scene.split("scene")[1]
            viewpoint = viewpoints[scene]
            categories = []
            bboxes = []
            if os.path.isdir(scene_path):
                for file in os.listdir(scene_path):
                    if file == f"{viewpoint}.npz":
                        print("Found npz file")
                        file_path = os.path.join(scene_path, file)
                        with np.load(file_path) as data:
                            print("processing scene: ", scene)
                            instances_semantic = data['instances_semantic']
                            instances_objects = data['instances_objects']
                            unique_instances = np.unique(instances_objects)
                            for instance_id in unique_instances:
                                if instance_id == 0:
                                    continue
                                instance_mask = (instances_objects == instance_id)
                                category_id = np.unique(instances_semantic[instance_mask])[0]
                                bbox = convert_masks_to_boxes(instance_mask)
                                bbox = convert_from_x1y1x2y2_to_coco(bbox)
                                categories.append(category_id)
                                bboxes.append(bbox)
                        break

                    else:
                        print(f"{viewpoint}.npz not found")
                    
            
            labels["images"].append({
                "id": int(scene_nb),
                "file_name": f"{scene}.png"
            })

            if len(bboxes) == 0:
                print(f"No objects found in scene {scene}_{viewpoint}")
            

            for cat, box in zip(categories, bboxes):
                box = box.squeeze()
                area = float(box[2] * box[3])
                labels["annotations"].append({
                    "id": ann_id,
                    "image_id": int(scene_nb),
                    "category_id": int(cat),
                    "bbox": box.tolist(),
                    "area": area,  # width * height
                    "iscrowd": 0    
                })
                ann_id += 1
                
    for cat_name, cat_id in CLASS2ID.items():
        labels["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "object"
        })

    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)

def get_scenes_with_single_category(labels_file):
    scene_for_each_category = {}

    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    for ann in labels["annotations"]:
        category_id = ann["category_id"]
        scene_id = ann["image_id"]
        if category_id not in scene_for_each_category:
            scene_for_each_category[category_id] = [scene_id]
        else:
            scene_for_each_category[category_id].append(scene_id)

    scene_for_each_category = {k: v for k, v in scene_for_each_category.items() if len(v) == 1}

    return scene_for_each_category
        
def find_query_images(labels_file, query_set_file, dir):
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
        scene_name = scene.split(".")[0].split("_")[0]
        if scene_name not in query_scenes:
            scene_path = os.path.join(source_dir, scene)
            im = Image.open(scene_path)
            new_file_path = os.path.join(new_dir, scene)
            im.save(new_file_path)

def create_subset(dir, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    imgs = []
    for file in os.listdir(dir):
        imgs.append(file)
    
    selected_files = random.sample(imgs, 100)
    for file in selected_files:
        scene_path = os.path.join(dir, file)
        im = Image.open(scene_path)
        new_file_path = os.path.join(new_dir, file)
        im.save(new_file_path)



if __name__ == '__main__':

    dir = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real"
    create_labels_file(dir, "MGN/MGN_gt.json", test=False)

    '''
    dir = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real\\data_ifl_4\\mnt\\data1\\data_ifl_real\\scene212\\3.npz"

    data = np.load(dir)

    viewpts = get_viewpoints("MGN/MGN_images")
    print(viewpts["scene212"])

    instances_semantic = data['instances_semantic']
    instances_objects = data['instances_objects']
    plt.imshow(instances_objects)
    plt.show()
    plt.imshow(instances_semantic)
    plt.show()
    
    

    file = "MGN/MGN_gt.json"
    with open(file, 'r') as f:
        labels = json.load(f)
    anns = labels["images"]

    unique_imgs = set(anns[i]["id"] for i in range(len(anns)))
    print(len(unique_imgs))
    #mgn_create_queries("MGN/MGN_gt.json", "MGN/MGN_images", 5000, 5)
    
    
    # Create labels file
    dir = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real"
    create_labels_file(dir, "Test/MGN/MGN_gt_val.json", test=False)

    
    path = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real\\data_ifl_0\\mnt\\data1\\data_ifl_real\\scene0\\3.npz"
    
    data = np.load(path)
    objects_semantic = data['instances_objects']
    plt.imshow(objects_semantic)
    plt.show()
    
   
    
    # Load all images from the dataset
    dir = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real"
    load_all_images(dir, "MGN_images")
    
    # Create labels file
    create_labels_file(dir, "Test/MGN/MGN_gt_val.json", test=True)

    create_subset("MGN/MGN_images", "MGN/MGN_subset")


    
    find_query_images("MGN/MGN_labels.json", "Queries/MGN_query_set", "MGN/MGN_images")
    
    scenes = get_scenes_with_single_category("MGN/MGN_gt.json")
    scenes = [scene for sublist in scenes.values() for scene in sublist]
    
    create_test_images("MGN/MGN_images", "Test/MGN_test", scenes) 
    
    '''