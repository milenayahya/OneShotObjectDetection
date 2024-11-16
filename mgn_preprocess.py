import h5py
import numpy as np
import os
from PIL import Image
import json

def print_structure(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset content: {obj[:]}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")
        for sub_name, sub_item in obj.items():
            print(f"  Sub-item name: {sub_name}")
            if isinstance(sub_item, h5py.Dataset):
                print(f"  Dataset content: {sub_item[:]}")
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
        

if __name__ == '__main__':

    # Load all images from the dataset
    dir = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real"
    #load_all_images(dir, "MGN_images")


    # Create labels file
    create_labels_file(dir, "MGN_labels.json")

    '''
    cats = load_all_labels(dir)
    print(cats)
    print(len(cats))
    
 
    file_path = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real\\data_ifl_0\\mnt\\data1\\data_ifl_real\\scene0\\3_scene.hdf5"
    

    #file_path = "C:\\Users\\cm03009\\Downloads\\MetaGraspNet_Real (1)\\MetaGraspNet_Real\\data_ifl_real_0_588.zip\\scene578\\"

    #
    # file_path = "C:\\Users\\cm03009\\Downloads\\MetaGraspNetV2_Real\\data_ifl_0\\mnt\\data1\\data_ifl_real\\scene0\\3.npz"

    with h5py.File(file_path, 'r') as f:
    # List all groups and datasets in the file
       
        f.visititems(print_structure)
        objects_group = f['objects']
        for name, item in objects_group.items():
            print(f"Item name: {name}")
            if isinstance(item, h5py.Dataset):
                print(f"Dataset content: {item}")

            elif isinstance(item, h5py.Group):
                print(f"Group: {name}")
                for sub_name, sub_item in item.items():
                    print(f"  Sub-item name: {sub_name}")
                    if isinstance(sub_item, h5py.Dataset):
                        print(f"  Dataset content: {sub_item[:]}")
                    elif isinstance(sub_item, h5py.Group):
                        print(f"  Sub-group: {sub_name}")

   
    # Check for 'bboxes_loose' dataset
    if 'bboxes_loose' in objects_group:
        bboxes_loose = objects_group['bboxes_loose']
        print("Bounding Boxes (bboxes_loose):")
        print(bboxes_loose[:])
    else:
        print("No 'bboxes_loose' dataset found in 'objects' group.")
    

    '''