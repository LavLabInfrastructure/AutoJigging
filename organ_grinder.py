import json
import yaml

#profile template
#profiles = {
#     "brain": {
#         "organ": "brain", // name of organ
#         "jig_translate_y": 15, // distance the organ is shifted towards the bottom of the jig
#         "slicing_z": 2.5, // jig knife slot thickness
#         "x_wall": 15, // x tolerance of the jig
#         "y_wall": 5, // y tolerance of the jig
#         "z_wall": 5, // z tolerance of the jig
#         "pre_knife_space": 50, // distance from the top of the organ to the top of the jig, based on the size of knife
#         "post_knife_space": 5, // distance from the bottom of the organ to the bottom of the jig, allowing for the knife to pass through
#         "label": 1, // MR label of the mask
#         "iterations": 10000, // # of smoothing iterations
#         "reduction": 0.8, // dictaes the percentage of triangles in the mesh to be decimated
#         "scale": (1.02, 1.02, 1.02) // scale the mesh to original size post smoothing and decimation
#         "tumor_laterality": "L" // what side the tumor is on
#     },

def create_profile():
    """
    Interactive script to create an organ profile for AutoJigger with default values.
    """
    print("Welcome to the Autojigger Profile Creator!")
    print("This script will help you create a custom profile for the autojigger. Default values will be for the brain profile. Press Enter to use the default value.")
    print("How would you like to cut your organs today?\n")

    def get_input(prompt, default=None, cast_type=str, required=False, choices=None):
        """
        Helper function to get user input with optional default, type casting, and validation.
        If `required` is True, the user must provide a value.
        If `choices` is provided, the input must match one of the choices.
        """
        while True:
            user_input = input(f"{prompt}{f' (default: {default})' if default else ''}: ").strip()
            if user_input:
                try:
                    value = cast_type(user_input)
                    if choices and value not in choices:
                        print(f"Invalid input. Please choose from {choices}.")
                        continue
                    return value
                except ValueError:
                    print(f"Invalid input. Please enter a valid {cast_type.__name__}.")
            elif default is not None:
                if choices and default not in choices:
                    print(f"Invalid default value. Please choose from {choices}.")
                    continue
                return default
            elif required:
                print("This field is required. Please provide a value.")
            else:
                return None

    organ = get_input("Organ name", required=True)
    jig_translate_y = get_input("Jig translate Y (distance the organ is shifted towards the bottom of the jig)", 15, float)
    slicing_z = get_input("Slicing Z (knife slot thickness)", 2.5, float)
    x_wall = get_input("X wall tolerance", 15, float)
    y_wall = get_input("Y wall tolerance", 5, float)
    z_wall = get_input("Z wall tolerance", 5, float)
    pre_knife_space = get_input("Pre-knife space distance", 50, float)
    post_knife_space = get_input("Post-knife space distance", 5, float)
    label = get_input("MR label of the mask", 1, int)
    iterations = get_input("Number of smoothing iterations", 10000, int)
    reduction = get_input("Reduction percentage", 0.8, float)
    scale_x = get_input("Scale factor for X", 1.02, float)
    scale_y = get_input("Scale factor for Y", 1.02, float)
    scale_z = get_input("Scale factor for Z", 1.02, float)
    laterality = get_input("Tumor laterality (optional, choose 'L' or 'R')", None, choices=["L", "R"])

    profile = {
        organ: {
            "organ": organ,
            "jig_translate_y": jig_translate_y,
            "slicing_z": slicing_z,
            "x_wall": x_wall,
            "y_wall": y_wall,
            "z_wall": z_wall,
            "pre_knife_space": pre_knife_space,
            "post_knife_space": post_knife_space,
            "label": label,
            "iterations": iterations,
            "reduction": reduction,
            "scale": (scale_x, scale_y, scale_z)
        }
    }

    if laterality:
        profile[organ]["tumor_laterality"] = laterality

    file_format = get_input("Save as JSON, YAML, or Python? (Enter 'json', 'yaml', or 'py')", "json").lower()

    if file_format == "json":
        file_name = f"{organ}_profile.json"
    elif file_format == "yaml":
        file_name = f"{organ}_profile.yaml"
    elif file_format == "py":
        file_name = f"{organ}_profile.py"
    else:
        print("Invalid file format. Profile not saved.")
        return

    try:
        if file_format == "json":
            with open(file_name, "w", encoding="utf-8") as json_file:
                json.dump(profile, json_file, indent=4)
            print(f"Profile saved to {file_name} as a .json.")
        elif file_format == "yaml":
            with open(file_name, "w", encoding="utf-8") as yaml_file:
                yaml.dump(profile, yaml_file)
            print(f"Profile saved to {file_name} as a .yaml.")
        elif file_format == "py":
            with open(file_name, "w", encoding="utf-8") as py_file:
                py_file.write("profiles = ")
                py_file.write(repr(profile)) 
                #py_file.write(json.dumps(profile, indent=4)) looks prettier but scale is list not tuple
            print(f"Profile saved to {file_name} as a .py.")
        else:
            print("Invalid file format. Profile not saved.")
    except Exception as e:
        print(f"An error occurred while saving the profile: {e}")

if __name__ == "__main__":
    create_profile()
