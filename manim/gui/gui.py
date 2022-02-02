from __future__ import annotations

from pathlib import Path

try:
    import dearpygui.dearpygui as dpg

    dearpygui_imported = True
except ImportError:
    dearpygui_imported = False


from .. import __version__, config
from ..utils.module_ops import scene_classes_from_file

if dearpygui_imported:
    dpg.create_context()
    window = dpg.generate_uuid()


def configure_pygui(renderer, widgets, update=True):
    if not dearpygui_imported:
        raise RuntimeError("Attempted to use DearPyGUI when it isn't imported.")
    if update:
        dpg.delete_item(window)
    else:
        dpg.create_viewport()
        dpg.setup_dearpygui()
        dpg.show_viewport()

    dpg.set_viewport_title(title=f"Manim Community v{__version__}")
    dpg.set_viewport_width(1015)
    dpg.set_viewport_height(540)

    def rerun_callback(sender, data):
        renderer.scene.queue.put(("rerun_gui", [], {}))

    def continue_callback(sender, data):
        renderer.scene.queue.put(("exit_gui", [], {}))

    def scene_selection_callback(sender, data):
        config["scene_names"] = (dpg.get_value(sender),)
        renderer.scene.queue.put(("rerun_gui", [], {}))

    scene_classes = scene_classes_from_file(Path(config["input_file"]), full_list=True)
    scene_names = [scene_class.__name__ for scene_class in scene_classes]

    with dpg.window(
        id=window,
        label="Manim GUI",
        pos=[config["gui_location"][0], config["gui_location"][1]],
        width=1000,
        height=500,
    ):
        dpg.set_global_font_scale(2)
        dpg.add_button(label="Rerun", callback=rerun_callback)
        dpg.add_button(label="Continue", callback=continue_callback)
        dpg.add_combo(
            label="Selected scene",
            items=scene_names,
            callback=scene_selection_callback,
            default_value=config["scene_names"][0],
        )
        dpg.add_separator()
        if len(widgets) != 0:
            with dpg.collapsing_header(
                label=f"{config['scene_names'][0]} widgets",
                default_open=True,
            ):
                for widget_config in widgets:
                    widget_config_copy = widget_config.copy()
                    name = widget_config_copy["name"]
                    widget = widget_config_copy["widget"]
                    if widget != "separator":
                        del widget_config_copy["name"]
                        del widget_config_copy["widget"]
                        getattr(dpg, f"add_{widget}")(label=name, **widget_config_copy)
                    else:
                        dpg.add_separator()

    if not update:
        dpg.start_dearpygui()
