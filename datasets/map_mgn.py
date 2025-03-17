ID2CLASS = {
    0: "food",
    1: "kitchenware",
    2: "tool",
    3: "model",
    4: "personal_care",
    5: "storage",
    6: "household_item"

}

ID2COLOR = {
    0: (1, 0, 0),       # Red for food
    1: (0, 1, 0),       # Green for kitchenware
    2: (0, 0, 1),       # Blue for tool
    3: (1, 1, 0),     # Yellow for model
    4: (1, 0, 1),     # Magenta for personal care
    5: (0, 1, 1),     # Cyan for storage
    6: (128/255, 0, 128/255)      # Purple for household item
}


CLASS2ID = {v: k for k, v in ID2CLASS.items()}

CAT_TO_SUPERCAT = {
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