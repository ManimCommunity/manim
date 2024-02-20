# manim.utils.quick_setup for ipython_magics.py %manim --quick-setup [--accept-default]
# ~/manim/manim/_config/default.cfg

import json
import os
import re
import sys

from manim import config
from manim.utils.color import manim_colors

# Global Variables
projectName = "media"  # Default Project Folder Name
# Allow any of these "formats" for boolean options
true_types = ["true", "yes", "y", "1"]
false_types = ["false", "no", "n", "0"]

config_data = {}


def project_setup():
    # Project Structure Phase
    print("==================")
    print("Project Structure:")
    print("==================")

    # The following code may be reduced by using a whilte loop
    # over a lit of attributes ["video_dir","partial_movie_dir", etc ...]

    # Get project folder name
    config_data["Project Name"] = prompt_project_name()
    config_data["config.media_dir"] = config_data["Project Name"]

    # prompt user for video_dir folder name (relative to project folder)
    config_data["config.video_dir"] = prompt_dir_config_attribute("video_dir")
    # prompt user for partial_movie_dir folder name (relative to project folder)
    config_data["config.partial_movie_dir"] = prompt_dir_config_attribute(
        "partial_movie_dir"
    )

    # prompt user whether to save video sections
    config_data["config.save_sections"] = prompt_bool_config_attribute("save_sections")
    if config_data["config.save_sections"]:
        # if True, prompt user for sections_dir folder name (relative to project folder)
        config_data["config.sections_dir"] = prompt_dir_config_attribute("sections_dir")

    # prompt user for image_dir folder name (relative to project folder)
    config_data["config.images_dir"] = prompt_dir_config_attribute("images_dir")
    # prompt user for tex_dir folder name (relative to project folder)
    config_data["config.tex_dir"] = prompt_dir_config_attribute("tex_dir")
    # prompt user for text_dir folder name (relative to project folder)
    config_data["config.text_dir"] = prompt_dir_config_attribute("text_dir")

    # prompt user whether to save media for Jupyter Lab/Notebooks rendering
    config_data["config.media_embed"] = prompt_bool_config_attribute("media_embed")

    # Workflow Settings Phase
    print("==================")
    print("Workflow Settings:")
    print("==================")

    verbosity_level_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    verbosity_level = input("config.verbosity = ")

    if verbosity_level.strip() == "":
        # config.verbosity = "INFO" # redundant but explicit
        pass  # set default verbosity level
    else:
        if not verbosity_level.upper() in verbosity_level_options:
            raise NameError(
                f"{verbosity_level} is not one of {verbosity_level_options}"
            )
        else:
            config.verbosity = verbosity_level.upper()
    config_data["config.verbosity"] = config.verbosity

    # Prompt user for enabling gui
    config_data["config.enable_gui"] = prompt_bool_config_attribute("enable_gui")

    # Media Settings Phase
    print("===============")
    print("Media Settings:")
    print("===============")

    # Prompt user for media background color
    config_data["config.background_color"] = prompt_background_color_config_attribute()

    # The following code may be reduced by using a whilte loop
    # over a lit of attributes ["media_width","frame_width", etc ...]

    # Prompt user for media width % # default config.media_width = 60%
    media_width_input = input("config.media_width (%) = ")  # must be str or falsy
    # Process media_width_input
    if media_width_input.strip() == "":
        # accept default
        pass
    else:
        if not is_percent(media_width_input):
            raise ValueError(
                f"Expected percentage input including '%' but received type({media_width_input}) = {type(media_width_input)}"
            )
        # check if media_input without % symbol is a number
        _media_width_input = media_width_input.replace("%", "", 1)
        if is_number(_media_width_input):
            # check if media_width_input is within bounds
            if float(_media_width_input) > 0 and float(_media_width_input) <= 100:
                config["media_width"] = media_width_input
            else:
                raise ValueError(
                    f"{media_width_input} is not within bounds of [0, 100]%"
                )
        else:
            raise TypeError(
                f"{media_width_input} is not a number representing a percentage"
            )

    config_data["config.media_width"] = config.media_width

    # Prompt user for frame width (float) # default config.frame_width = 14.222222222222221
    config_data["config.frame_width"] = prompt_number_config_attribute(
        "frame_width", float(0), float("inf")
    )  # use -1 for infinity

    # Prompt user for frame height (float) # default config.frame_height = 8.0
    config_data["config.frame_height"] = prompt_number_config_attribute(
        "frame_height", float(0), float("inf")
    )  # use -1 for infinity

    # https://www.omnicalculator.com/other/video-frame-size
    # https://en.wikipedia.org/wiki/Display_resolution

    # Prompt user for pixel width (int) # default config.pixel_width = 1920
    config_data["config.pixel_width"] = prompt_number_config_attribute(
        "pixel_width", 0, float("inf")
    )  # use -1 for infinity

    # Prompt user for pixel height (int) # default config.pixel_height = 1080
    config_data["config.pixel_heights"] = prompt_number_config_attribute(
        "pixel_height", 0, float("inf")
    )  # use -1 for infinity

    # Prompt user for frame rate (float) (0.0 fps - 60.0fps)
    config_data["config.frame_rate"] = prompt_number_config_attribute(
        "frame_rate", float(0), float(60)
    )

    # Extra Settings Phase
    print("===============")
    print("Extra Settings:")
    print("===============")

    import_manimpango = input("Import manimpango fonts library?")
    if import_manimpango.lower() in false_types:
        config_data["manimpango"] = False
    elif import_manimpango.strip() == "" or import_manimpango in true_types:
        import manimpango

        print("\u2713 import manimpango")
        print(
            "For a comprehensive list of available fonts invoke `manimpango.list_fonts()`"
        )
        config_data["manimpango"] = True
    else:
        raise NameError(
            f"Expected {true_types} or {false_types} but received type({import_manimpango}) = {type(import_manimpango)}"
        )

    save_JSON = input("Save config file as `.json`?")
    if save_JSON.lower() in false_types:
        pass
    elif save_JSON.strip() == "" or save_JSON in true_types:
        import shutil
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        json.dump(config_data, temp_file)
        temp_file.flush()
        print(temp_file.name)
        print(tempfile.gettempdir())
        temp_file_path = os.path.join(tempfile.gettempdir(), temp_file.name)
        # Save config.json file
        dir = os.path.join(os.getcwd(), config_data["Project Name"])
        # Make the directory if it does not yet exist
        os.mkdir(dir)
        path = os.path.join(dir, "config.json")
        shutil.copy(temp_file_path, path)
        os.remove(temp_file_path)
        print(f"\u2713 JSON config file ready @ {path}")
    else:
        raise NameError(
            f"Expected {true_types} or {false_types} but received type({save_JSON}) = {type(save_JSON)}"
        )

    return


def prompt_project_name():
    # promp user for project folder name in which to render manim output
    # if user specifies no project name, accept default
    global projectName
    projectName = input("Enter project name:")

    if projectName.strip() == "":
        print('No project name specified; default folder name: "media"')
        projectName = "media"

    validate_folder_name(projectName)
    config.media_dir = projectName
    return projectName


def validate_folder_name(projectName):
    # Check that user entered an accepted project name

    # Check for non-string input
    if not isinstance(projectName, str):
        raise TypeError(f"Expecting a string as input, not {type(projectName)}")

    # check that user input contains only alphabetic characters, whitespace, hyphens and underscores
    # https://stackoverflow.com/questions/38265411/regular-expression-for-alphanumeric-hyphen-and-underscore-without-any-space
    if not re.search(r"^[a-zA-Z0-9\s\/_-]*$", projectName):
        raise ValueError(
            f"Folder names [{projectName}] may consist only of alphanumerics, whitespace, hyphens, underscore & forwardslash characters"
        )


def prompt_bool_config_attribute(attribute):
    # Allow any of these "formats" for boolean options
    # true_types = ["true", "yes", "y", "1"]
    # false_types = ["false", "no", "n", "0"]

    attribute_value = input(f"config.{attribute} (bool) = ")

    # Convert input string to boolean
    if attribute_value.strip() == "":
        return config[f"{attribute}"]  # use default config value
    elif attribute_value.lower() in true_types:
        config[f"{attribute}"] = True
    elif attribute_value.lower() in false_types:
        config[f"{attribute}"] = False
    else:
        # Any other input is forbidden
        raise TypeError(
            f"Expected boolean type, but type({attribute}) = {type(attribute_value)}"
        )

    return config[f"{attribute}"]


def prompt_dir_config_attribute(attribute):
    # Find the current working directory in which the
    # python jupyter notebook file .ipynb is contained

    directory = os.getcwd()
    FOLDER = os.path.join(directory, projectName)

    attribute_value = input(f"config.{attribute} = {FOLDER}/")

    # if user input is empty, accept default
    if attribute_value == "":
        return config[f"{attribute}"]  # use default config value

    # validate directory path
    validate_folder_name(attribute_value)

    config[f"{attribute}"] = os.path.join(FOLDER, attribute_value)
    return config[f"{attribute}"]


def prompt_background_color_config_attribute():
    # Colors included in the global name space
    # https://github.com/ManimCommunity/manim/blob/main/manim/utils/color/manim_colors.py
    backgroundColor = input("Background Colour (BLACK/WHITE/etc...):")
    # print(manim_colors._all_manim_colors)
    # print(type(vars(manim_colors)))
    if backgroundColor.strip() == "":
        print("Default Background Color: BLACK")
        # config.background_color = "BLACK" # set default background color
        return "BLACK"
    else:
        if not backgroundColor.upper() in vars(manim_colors):
            # Iterate over global variables and print keys which contain "ManimColor" as substring
            # ManimColor takes a hex value starting with #
            # thus all color names can be obtained as keys
            # from their values, which contain # as a substring
            colors = []
            for key in list(vars(manim_colors).keys()):
                if "#" in str(vars(manim_colors)[key]):
                    colors.append(str(key))
            colors = [
                x for x in colors if x != "config"
            ]  # exclude config global variable from list
            print(colors)
            print(
                "@ https://docs.manim.community/en/stable/reference/manim.utils.color.manim_colors.html"
            )
            raise NameError(f"{backgroundColor} is not a valid color")
        else:
            config.background_color = (
                backgroundColor  # accept user input as config.background_color
            )
    return backgroundColor


def prompt_number_config_attribute(attribute, min, max):
    # Get input from user
    numberInput = input(f"config.{attribute} [{min}, {max}] = ")
    if numberInput.strip() == "":
        pass  # accept default
    elif is_number(numberInput):
        # check whether value is within bounds or -1
        if float(numberInput) > min and float(numberInput) <= max:
            # check whether value is a float
            if is_float(numberInput):
                config[f"{attribute}"] = float(numberInput)
            else:
                config[f"{attribute}"] = int(numberInput)
        elif numberInput == "-1":  # for infinity
            config[f"{attribute}"] = int(numberInput)
        else:
            raise ValueError(f"{numberInput} is not within bounds of [{min}, {max}]")
    else:
        raise TypeError(
            f"Expected int or float input, but received type({numberInput}) = {type(numberInput)}"
        )
    return config[f"{attribute}"]


def is_number(number):
    return number.replace(".", "", 1).replace("-", "", 1).isnumeric()


def is_float(number):
    if "." in number:
        return True
    else:
        return False


def is_percent(number):
    if "%" in number:
        return True
    else:
        return False
