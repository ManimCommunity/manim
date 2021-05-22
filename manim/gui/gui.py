from pathlib import Path

import dearpygui.core
import dearpygui.demo
import dearpygui.simple

from .. import config
from ..utils.module_ops import scene_classes_from_file


def configure_pygui(renderer, widgets, update=True):
    if update:
        dearpygui.core.delete_item("Manim GUI")

    def rerun_callback(sender, data):
        renderer.scene.queue.put(("rerun_gui", [], {}))

    def continue_callback(sender, data):
        renderer.scene.queue.put(("exit_gui", [], {}))

    def scene_selection_callback(sender, data):
        config["scene_names"] = (dearpygui.core.get_value(sender),)
        renderer.scene.queue.put(("rerun_gui", [], {}))

    scene_classes = scene_classes_from_file(Path(config["input_file"]), full_list=True)
    scene_names = [scene_class.__name__ for scene_class in scene_classes]

    with dearpygui.simple.window(
        "Manim GUI", x_pos=40, y_pos=800, width=1000, height=500
    ):
        dearpygui.core.set_global_font_scale(2)
        # dearpygui.core.add_additional_font(
        #     "roboto", "../../../../.local/share/fonts/Roboto-Black.ttf", 30
        # )
        # dearpygui.core.set_font("roboto", 30)
        dearpygui.core.add_button("Rerun", callback=rerun_callback)
        dearpygui.core.add_button("Continue", callback=continue_callback)
        dearpygui.core.add_combo(
            "Selected scene",
            items=scene_names,
            callback=scene_selection_callback,
            default_value=config["scene_names"][0],
        )
        dearpygui.core.add_separator()
        if len(widgets) != 0:
            with dearpygui.simple.collapsing_header(
                f"{config['scene_names'][0]} widgets", default_open=True
            ):
                for widget_config in widgets:
                    widget_config_copy = widget_config.copy()
                    name = widget_config_copy["name"]
                    widget = widget_config_copy["widget"]
                    if widget != "separator":
                        del widget_config_copy["name"]
                        del widget_config_copy["widget"]
                        getattr(dearpygui.core, f"add_{widget}")(
                            name, **widget_config_copy
                        )
                    else:
                        dearpygui.core.add_separator()
    # dearpygui.demo.show_demo()
    if not update:
        dearpygui.core.start_dearpygui()
