import inspect
import random
import warnings
import platform
import copy

from tqdm import tqdm as ProgressDisplay
import numpy as np

from ..animation.animation import Animation
from ..animation.transform import MoveToTarget, ApplyMethod
from ..camera.camera import Camera
from ..constants import *
from ..config import camera_config, file_writer_config
from ..container import Container
from ..logger import logger
from ..mobject.mobject import Mobject
from ..scene.scene_file_writer import SceneFileWriter
from ..utils.iterables import list_update
from ..utils.hashing import get_hash_from_play_call, get_hash_from_wait_call


class Scene(Container):
    """A Scene is the canvas of your animation.

    All of your own named Scenes will be subclasses of Scene, or other named
    scenes.

    Examples
    --------
    Override the construct() method to tell Manim what should go on in the
    Scene.

    .. code-block:: python

        class MyScene(Scene):
            def construct(self):
                self.play(
                    Write(Text("Hello World!"))
                )

    Some important variables to note are:
        camera: The camera object to be used for the scene.
        file_writer : The object that writes the animations in the scene to a video file.
        mobjects : The list of mobjects present in the scene.
        foreground_mobjects : List of mobjects explicitly in the foreground.
        num_plays : Number of play() functions in the scene.
        time: time elapsed since initialisation of scene.
        random_seed: The seed with which all random operations are done.

    """

    CONFIG = {
        "camera_class": Camera,
        "skip_animations": False,
        "always_update_mobjects": False,
        "random_seed": 0,
    }

    def __init__(self, **kwargs):
        Container.__init__(self, **kwargs)
        self.camera = self.camera_class(**camera_config)
        self.file_writer = SceneFileWriter(self, **file_writer_config,)
        self.play_hashes_list = []
        self.mobjects = []
        # TODO, remove need for foreground mobjects
        self.foreground_mobjects = []
        self.num_plays = 0
        self.time = 0
        self.original_skipping_status = file_writer_config["skip_animations"]
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        self.setup()
        try:
            self.construct()
        except EndSceneEarlyException:
            pass
        self.tear_down()
        # We have to reset these settings in case of multiple renders.
        file_writer_config["skip_animations"] = False
        self.original_skipping_status = file_writer_config["skip_animations"]
        self.file_writer.finish()
        self.print_end_message()

    def setup(self):
        """
        This is meant to be implemented by any scenes which
        are comonly subclassed, and have some common setup
        involved before the construct method is called.
        """
        pass

    def tear_down(self):
        """
        This is meant to be implemented by any scenes which
        are comonly subclassed, and have some common method
        to be invoked before the scene ends.
        """
        pass

    def construct(self):
        """
        The primary method for constructing (i.e adding content to)
        the Scene.
        """
        pass  # To be implemented in subclasses

    def __str__(self):
        return self.__class__.__name__

    def print_end_message(self):
        """
        Used internally to print the number of
        animations played after the scene ends,
        as well as the name of the scene rendered
        (useful when using the `-a` option).
        """
        logger.info(f"Rendered {str(self)}\nPlayed {self.num_plays} animations")

    def set_variables_as_attrs(self, *objects, **newly_named_objects):
        """
        This method is slightly hacky, making it a little easier
        for certain methods (typically subroutines of construct)
        to share local variables.
        """
        caller_locals = inspect.currentframe().f_back.f_locals
        for key, value in list(caller_locals.items()):
            for o in objects:
                if value is o:
                    setattr(self, key, value)
        for key, value in list(newly_named_objects.items()):
            setattr(self, key, value)
        return self

    def get_attrs(self, *keys):
        """
        Gets attributes of a scene given the attribute's identifier/name.

        Parameters
        ----------
        *keys : str
            Name(s) of the argument(s) to return the attribute of.

        Returns
        -------
        list
            List of attributes of the passed identifiers.
        """
        return [getattr(self, key) for key in keys]

    def get_frame(self):
        """
        Gets the current frame as NumPy array.

        Returns
        -------
        np.array
            NumPy array of pixel values of each pixel in screen.
            The shape of the array is height x width x 3
        """
        return np.array(self.camera.get_pixel_array())

    def update_frame(  # TODO Description in Docstring
        self,
        mobjects=None,
        background=None,
        include_submobjects=True,
        ignore_skipping=True,
        **kwargs,
    ):
        """Update the frame.

        Parameters
        ----------
        mobjects: list, optional
            list of mobjects

        background: np.ndarray, optional
            Pixel Array for Background.

        include_submobjects: bool, optional

        ignore_skipping : bool, optional

        **kwargs

        """
        if file_writer_config["skip_animations"] and not ignore_skipping:
            return
        if mobjects is None:
            mobjects = list_update(self.mobjects, self.foreground_mobjects,)
        if background is not None:
            self.camera.set_pixel_array(background)
        else:
            self.camera.reset()

        kwargs["include_submobjects"] = include_submobjects
        self.camera.capture_mobjects(mobjects, **kwargs)

    def freeze_background(self):
        self.update_frame()
        self.camera = Camera(self.get_frame())
        self.clear()

    ###

    def update_mobjects(self, dt):
        """
        Begins updating all mobjects in the Scene.

        Parameters
        ----------
        dt: int or float
            Change in time between updates. Defaults (mostly) to 1/frames_per_second
        """
        for mobject in self.mobjects:
            mobject.update(dt)

    def should_update_mobjects(self):
        """
        Returns True if any mobject in Scene is being updated
        or if the scene has always_update_mobjects set to true.

        Returns
        -------
            bool
        """
        return self.always_update_mobjects or any(
            [mob.has_time_based_updater() for mob in self.get_mobject_family_members()]
        )

    ###

    def increment_time(self, d_time):
        """
        Increments the time elapsed after intialisation of scene by
        passed "d_time".

        Parameters
        ----------
        d_time : int or float
            Time in seconds to increment by.
        """
        self.time += d_time

    ###

    def get_top_level_mobjects(self):
        """
        Returns all mobjects which are not submobjects.

        Returns
        -------
        list
            List of top level mobjects.
        """
        # Return only those which are not in the family
        # of another mobject from the scene
        mobjects = self.get_mobjects()
        families = [m.get_family() for m in mobjects]

        def is_top_level(mobject):
            num_families = sum([(mobject in family) for family in families])
            return num_families == 1

        return list(filter(is_top_level, mobjects))

    def get_mobject_family_members(self):
        """
        Returns list of family-members of all mobjects in scene.
        If a Circle() and a VGroup(Rectangle(),Triangle()) were added,
        it returns not only the Circle(), Rectangle() and Triangle(), but
        also the VGroup() object.

        Returns
        -------
        list
            List of mobject family members.
        """
        return self.camera.extract_mobject_family_members(self.mobjects)

    def add(self, *mobjects):
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.

        Parameters
        ---------
        *mobjects : Mobject
            Mobjects to add.

        Returns
        -------
        Scene
            The same scene after adding the Mobjects in.

        """
        mobjects = [*mobjects, *self.foreground_mobjects]
        self.restructure_mobjects(to_remove=mobjects)
        self.mobjects += mobjects
        return self

    def add_mobjects_among(self, values):
        """
        This is meant mostly for quick prototyping,
        e.g. to add all mobjects defined up to a point,
        call self.add_mobjects_among(locals().values())
        """
        self.add(*filter(lambda m: isinstance(m, Mobject), values))
        return self

    def add_mobjects_from_animations(self, animations):

        curr_mobjects = self.get_mobject_family_members()
        for animation in animations:
            # Anything animated that's not already in the
            # scene gets added to the scene
            mob = animation.mobject
            if mob not in curr_mobjects:
                self.add(mob)
                curr_mobjects += mob.get_family()

    def remove(self, *mobjects):
        """
        Removes mobjects in the passed list of mobjects
        from the scene and the foreground, by removing them
        from "mobjects" and "foreground_mobjects"

        Parameters
        ----------
        *mobjects : Mobject
            The mobjects to remove.
        """
        for list_name in "mobjects", "foreground_mobjects":
            self.restructure_mobjects(mobjects, list_name, False)
        return self

    def restructure_mobjects(
        self, to_remove, mobject_list_name="mobjects", extract_families=True
    ):
        """
        tl:wr
            If your scene has a Group(), and you removed a mobject from the Group,
            this dissolves the group and puts the rest of the mobjects directly
            in self.mobjects or self.foreground_mobjects.

        In cases where the scene contains a group, e.g. Group(m1, m2, m3), but one
        of its submobjects is removed, e.g. scene.remove(m1), the list of mobjects
        will be edited to contain other submobjects, but not m1, e.g. it will now
        insert m2 and m3 to where the group once was.

        Parameters
        ----------
        to_remove : Mobject
            The Mobject to remove.

        mobject_list_name : str, optional
            The list of mobjects ("mobjects", "foreground_mobjects" etc) to remove from.

        extract_families : bool, optional
            Whether the mobject's families should be recursively extracted.

        Returns
        -------
        Scene
            The Scene mobject with restructured Mobjects.
        """
        if extract_families:
            to_remove = self.camera.extract_mobject_family_members(to_remove)
        _list = getattr(self, mobject_list_name)
        new_list = self.get_restructured_mobject_list(_list, to_remove)
        setattr(self, mobject_list_name, new_list)
        return self

    def get_restructured_mobject_list(self, mobjects, to_remove):
        """
        Given a list of mobjects and a list of mobjects to be removed, this
        filters out the removable mobjects from the list of mobjects.

        Parameters
        ----------

        mobjects : list
            The Mobjects to check.

        to_remove : list
            The list of mobjects to remove.

        Returns
        -------
        list
            The list of mobjects with the mobjects to remove removed.
        """

        new_mobjects = []

        def add_safe_mobjects_from_list(list_to_examine, set_to_remove):
            for mob in list_to_examine:
                if mob in set_to_remove:
                    continue
                intersect = set_to_remove.intersection(mob.get_family())
                if intersect:
                    add_safe_mobjects_from_list(mob.submobjects, intersect)
                else:
                    new_mobjects.append(mob)

        add_safe_mobjects_from_list(mobjects, set(to_remove))
        return new_mobjects

    # TODO, remove this, and calls to this
    def add_foreground_mobjects(self, *mobjects):
        """
        Adds mobjects to the foreground, and internally to the list
        foreground_mobjects, and mobjects.

        Parameters
        ----------
        *mobjects : Mobject
            The Mobjects to add to the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobjects added.
        """
        self.foreground_mobjects = list_update(self.foreground_mobjects, mobjects)
        self.add(*mobjects)
        return self

    def add_foreground_mobject(self, mobject):
        """
        Adds a single mobject to the foreground, and internally to the list
        foreground_mobjects, and mobjects.

        Parameters
        ----------
        mobject : Mobject
            The Mobject to add to the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobject added.
        """
        return self.add_foreground_mobjects(mobject)

    def remove_foreground_mobjects(self, *to_remove):
        """
        Removes mobjects from the foreground, and internally from the list
        foreground_mobjects.

        Parameters
        ----------
        *to_remove : Mobject
            The mobject(s) to remove from the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobjects removed.
        """
        self.restructure_mobjects(to_remove, "foreground_mobjects")
        return self

    def remove_foreground_mobject(self, mobject):
        """
        Removes a single mobject from the foreground, and internally from the list
        foreground_mobjects.

        Parameters
        ----------
        mobject : Mobject
            The mobject to remove from the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobject removed.
        """
        return self.remove_foreground_mobjects(mobject)

    def bring_to_front(self, *mobjects):
        """
        Adds the passed mobjects to the scene again,
        pushing them to he front of the scene.

        Parameters
        ----------
        *mobjects : Mobject
            The mobject(s) to bring to the front of the scene.

        Returns
        ------
        Scene
            The Scene, with the mobjects brought to the front
            of the scene.
        """
        self.add(*mobjects)
        return self

    def bring_to_back(self, *mobjects):
        """
        Removes the mobject from the scene and
        adds them to the back of the scene.

        Parameters
        ----------
        *mobjects : Mobject
            The mobject(s) to push to the back of the scene.

        Returns
        ------
        Scene
            The Scene, with the mobjects pushed to the back
            of the scene.
        """
        self.remove(*mobjects)
        self.mobjects = list(mobjects) + self.mobjects
        return self

    def clear(self):
        """
        Removes all mobjects present in self.mobjects
        and self.foreground_mobjects from the scene.

        Returns
        ------
        Scene
            The Scene, with all of its mobjects in
            self.mobjects and self.foreground_mobjects
            removed.
        """
        self.mobjects = []
        self.foreground_mobjects = []
        return self

    def get_mobjects(self):
        """
        Returns all the mobjects in self.mobjects

        Returns
        ------
        list
            The list of self.mobjects .
        """
        return list(self.mobjects)

    def get_mobject_copies(self):
        """
        Returns a copy of all mobjects present in
        self.mobjects .

        Returns
        ------
        list
            A list of the copies of all the mobjects
            in self.mobjects
        """
        return [m.copy() for m in self.mobjects]

    def get_moving_mobjects(self, *animations):
        """
        Gets all moving mobjects in the passed animation(s).

        Parameters
        ----------
        *animations : Animation
            The animations to check for moving mobjects.

        Returns
        ------
        list
            The list of mobjects that could be moving in
            the Animation(s)
        """
        # Go through mobjects from start to end, and
        # as soon as there's one that needs updating of
        # some kind per frame, return the list from that
        # point forward.
        animation_mobjects = [anim.mobject for anim in animations]
        mobjects = self.get_mobject_family_members()
        for i, mob in enumerate(mobjects):
            update_possibilities = [
                mob in animation_mobjects,
                len(mob.get_family_updaters()) > 0,
                mob in self.foreground_mobjects,
            ]
            if any(update_possibilities):
                return mobjects[i:]
        return []

    def get_time_progression(
        self, run_time, n_iterations=None, override_skip_animations=False
    ):
        """
        You will hardly use this when making your own animations.
        This method is for Manim's internal use.

        Returns a CommandLine ProgressBar whose fill_time
        is dependent on the run_time of an animation,
        the iterations to perform in that animation
        and a bool saying whether or not to consider
        the skipped animations.

        Parameters
        ----------
        run_time: float
            The run_time of the animation.

        n_iterations: int, optional
            The number of iterations in the animation.

        override_skip_animations: bool, optional
            Whether or not to show skipped animations in the progress bar.

        Returns
        ------
        ProgressDisplay
            The CommandLine Progress Bar.
        """
        if file_writer_config["skip_animations"] and not override_skip_animations:
            times = [run_time]
        else:
            step = 1 / self.camera.frame_rate
            times = np.arange(0, run_time, step)
        time_progression = ProgressDisplay(
            times,
            total=n_iterations,
            leave=file_writer_config["leave_progress_bars"],
            ascii=True if platform.system() == "Windows" else None,
            disable=not file_writer_config["progress_bar"],
        )
        return time_progression

    def get_run_time(self, animations):
        """
        Gets the total run time for a list of animations.

        Parameters
        ----------
        animations: list of Animation
            A list of the animations whose total
            run_time is to be calculated.

        Returns
        ------
        float
            The total run_time of all of the animations in the list.
        """

        return np.max([animation.run_time for animation in animations])

    def get_animation_time_progression(self, animations):
        """
        You will hardly use this when making your own animations.
        This method is for Manim's internal use.

        Uses get_time_progression to obtaina
        CommandLine ProgressBar whose fill_time is
        dependent on the qualities of the passed Animation,

        Parameters
        ----------
        animations : list of Animation
            The list of animations to get
            the time progression for.

        Returns
        ------
        ProgressDisplay
            The CommandLine Progress Bar.
        """
        run_time = self.get_run_time(animations)
        time_progression = self.get_time_progression(run_time)
        time_progression.set_description(
            "".join(
                [
                    "Animation {}: ".format(self.num_plays),
                    str(animations[0]),
                    (", etc." if len(animations) > 1 else ""),
                ]
            )
        )
        return time_progression

    def compile_play_args_to_animation_list(self, *args, **kwargs):
        """
        Each arg can either be an animation, or a mobject method
        followed by that methods arguments (and potentially follow
        by a dict of kwargs for that method).
        This animation list is built by going through the args list,
        and each animation is simply added, but when a mobject method
        s hit, a MoveToTarget animation is built using the args that
        follow up until either another animation is hit, another method
        is hit, or the args list runs out.

        Parameters
        ----------
        *args : Animation or method of a mobject, which is followed by that method's arguments

        **kwargs : any named arguments like run_time or lag_ratio.

        Returns
        -------
        list : list of animations with the parameters applied to them.
        """
        animations = []
        state = {
            "curr_method": None,
            "last_method": None,
            "method_args": [],
        }

        def compile_method(state):
            if state["curr_method"] is None:
                return
            mobject = state["curr_method"].__self__
            if state["last_method"] and state["last_method"].__self__ is mobject:
                animations.pop()
                # method should already have target then.
            else:
                mobject.generate_target()
            #
            if len(state["method_args"]) > 0 and isinstance(
                state["method_args"][-1], dict
            ):
                method_kwargs = state["method_args"].pop()
            else:
                method_kwargs = {}
            state["curr_method"].__func__(
                mobject.target, *state["method_args"], **method_kwargs
            )
            animations.append(MoveToTarget(mobject))
            state["last_method"] = state["curr_method"]
            state["curr_method"] = None
            state["method_args"] = []

        for arg in args:
            if isinstance(arg, Animation):
                compile_method(state)
                animations.append(arg)
            elif inspect.ismethod(arg):
                compile_method(state)
                state["curr_method"] = arg
            elif state["curr_method"] is not None:
                state["method_args"].append(arg)
            elif isinstance(arg, Mobject):
                raise Exception(
                    """
                    I think you may have invoked a method
                    you meant to pass in as a Scene.play argument
                """
                )
            else:
                raise Exception("Invalid play arguments")
        compile_method(state)

        for animation in animations:
            # This is where kwargs to play like run_time and rate_func
            # get applied to all animations
            animation.update_config(**kwargs)

        return animations

    def update_skipping_status(self):
        """
        This method is used internally to check if the current
        animation needs to be skipped or not. It also checks if
        the number of animations that were played correspond to
        the number of animations that need to be played, and
        raises an EndSceneEarlyException if they don't correspond.
        """

        if file_writer_config["from_animation_number"]:
            if self.num_plays == file_writer_config["from_animation_number"]:
                file_writer_config["skip_animations"] = False
        if file_writer_config["upto_animation_number"]:
            if self.num_plays >= file_writer_config["upto_animation_number"]:
                file_writer_config["skip_animations"] = True
                raise EndSceneEarlyException()

    def handle_caching_play(func):
        """
        Decorator that returns a wrapped version of func that will compute the hash of the play invocation.

        The returned function will act according to the computed hash: either skip the animation because it's already cached, or let the invoked function play normally.

        Parameters
        ----------
        func : Callable[[...], None]
            The play like function that has to be written to the video file stream. Take the same parameters as `scene.play`.
        """

        def wrapper(self, *args, **kwargs):
            self.revert_to_original_skipping_status()
            animations = self.compile_play_args_to_animation_list(*args, **kwargs)
            self.add_mobjects_from_animations(animations)
            if not file_writer_config["disable_caching"]:
                mobjects_on_scene = self.get_mobjects()
                hash_play = get_hash_from_play_call(
                    self.camera, animations, mobjects_on_scene
                )
                self.play_hashes_list.append(hash_play)
                if self.file_writer.is_already_cached(hash_play):
                    logger.info(
                        f"Animation {self.num_plays} : Using cached data (hash : {hash_play})"
                    )
                    file_writer_config["skip_animations"] = True
            else:
                hash_play = "uncached_{:05}".format(self.num_plays)
                self.play_hashes_list.append(hash_play)
            func(self, *args, **kwargs)

        return wrapper

    def handle_caching_wait(func):
        """
        Decorator that returns a wrapped version of func that will compute the hash of the wait invocation.

        The returned function will act according to the computed hash: either skip the animation because it's already cached, or let the invoked function play normally.

        Parameters
        ----------
        func : Callable[[...], None]
            The wait like function that has to be written to the video file stream. Take the same parameters as `scene.wait`.
        """

        def wrapper(self, duration=DEFAULT_WAIT_TIME, stop_condition=None):
            self.revert_to_original_skipping_status()
            if not file_writer_config["disable_caching"]:
                hash_wait = get_hash_from_wait_call(
                    self.camera, duration, stop_condition, self.get_mobjects()
                )
                self.play_hashes_list.append(hash_wait)
                if self.file_writer.is_already_cached(hash_wait):
                    logger.info(
                        f"Wait {self.num_plays} : Using cached data (hash : {hash_wait})"
                    )
                    file_writer_config["skip_animations"] = True
            else:
                hash_wait = "uncached_{:05}".format(self.num_plays)
                self.play_hashes_list.append(hash_wait)
            func(self, duration, stop_condition)

        return wrapper

    def handle_play_like_call(func):
        """
        This method is used internally to wrap the
        passed function, into a function that
        actually writes to the video stream.
        Simultaneously, it also adds to the number
        of animations played.

        Parameters
        ----------
        func : function
            The play() like function that has to be
            written to the video file stream.

        Returns
        -------
        function
            The play() like function that can now write
            to the video file stream.
        """

        def wrapper(self, *args, **kwargs):
            self.update_skipping_status()
            allow_write = not file_writer_config["skip_animations"]
            self.file_writer.begin_animation(allow_write)
            func(self, *args, **kwargs)
            self.file_writer.end_animation(allow_write)
            self.num_plays += 1

        return wrapper

    def begin_animations(self, animations):
        """
        This method begins the list of animations that is passed,
        and adds any mobjects involved (if not already present)
        to the scene again.

        Parameters
        ----------
        animations : list
            List of involved animations.

        """
        for animation in animations:
            # Begin animation
            animation.begin()

    def progress_through_animations(self, animations):
        """
        This method progresses through each animation
        in the list passed and and updates the frames as required.

        Parameters
        ----------
        animations : list
            List of involved animations.
        """
        # Paint all non-moving objects onto the screen, so they don't
        # have to be rendered every frame
        moving_mobjects = self.get_moving_mobjects(*animations)
        self.update_frame(excluded_mobjects=moving_mobjects)
        static_image = self.get_frame()
        last_t = 0
        for t in self.get_animation_time_progression(animations):
            dt = t - last_t
            last_t = t
            for animation in animations:
                animation.update_mobjects(dt)
                alpha = t / animation.run_time
                animation.interpolate(alpha)
            self.update_mobjects(dt)
            self.update_frame(moving_mobjects, static_image)
            self.add_frames(self.get_frame())

    def finish_animations(self, animations):
        """
        This function cleans up after the end
        of each animation in the passed list.

        Parameters
        ----------
        animations : list
            list of animations to finish.
        """
        for animation in animations:
            animation.finish()
            animation.clean_up_from_scene(self)
        self.mobjects_from_last_animation = [anim.mobject for anim in animations]
        if file_writer_config["skip_animations"]:
            # TODO, run this call in for each animation?
            self.update_mobjects(self.get_run_time(animations))
        else:
            self.update_mobjects(0)

    @handle_caching_play
    @handle_play_like_call
    def play(self, *args, **kwargs):
        """
        This method is used to prep the animations for rendering,
        apply the arguments and parameters required to them,
        render them, and write them to the video file.

        Parameters
        ----------
        *args : Animation or mobject with mobject method and params
        **kwargs : named parameters affecting what was passed in *args e.g run_time, lag_ratio etc.
        """
        if len(args) == 0:
            warnings.warn("Called Scene.play with no animations")
            return
        animations = self.compile_play_args_to_animation_list(*args, **kwargs)
        self.begin_animations(animations)
        self.progress_through_animations(animations)
        self.finish_animations(animations)

    def clean_up_animations(self, *animations):
        """
        This method cleans up and removes from the
        scene all the animations that were passed

        Parameters
        ----------
        *animations : Animation
            Animation to clean up.

        Returns
        -------
        Scene
            The scene with the animations
            cleaned up.

        """
        for animation in animations:
            animation.clean_up_from_scene(self)
        return self

    def get_mobjects_from_last_animation(self):
        """
        This method returns the mobjects from the previous
        played animation, if any exist, and returns an empty
        list if not.

        Returns
        --------
        list
            The list of mobjects from the previous animation.

        """
        if hasattr(self, "mobjects_from_last_animation"):
            return self.mobjects_from_last_animation
        return []

    def get_wait_time_progression(self, duration, stop_condition):
        """
        This method is used internally to obtain the CommandLine
        Progressbar for when self.wait() is called in a scene.

        Parameters
        ----------
        duration: int or float
            duration of wait time

        stop_condition : function
            The function which determines whether to continue waiting.

        Returns
        -------
        ProgressBar
            The CommandLine ProgressBar of the wait time

        """
        if stop_condition is not None:
            time_progression = self.get_time_progression(
                duration,
                n_iterations=-1,  # So it doesn't show % progress
                override_skip_animations=True,
            )
            time_progression.set_description(
                "Waiting for {}".format(stop_condition.__name__)
            )
        else:
            time_progression = self.get_time_progression(duration)
            time_progression.set_description("Waiting {}".format(self.num_plays))
        return time_progression

    @handle_caching_wait
    @handle_play_like_call
    def wait(self, duration=DEFAULT_WAIT_TIME, stop_condition=None):
        """
        This method is used to wait, and do nothing to the scene, for some
        duration.
        Updaters stop updating, nothing happens.

        Parameters
        ----------
        duration : float or int, optional
            The duration of wait time.
        stop_condition :
            A function that determines whether to stop waiting or not.

        Returns
        -------
        Scene
            The scene, after waiting.
        """
        self.update_mobjects(dt=0)  # Any problems with this?
        if self.should_update_mobjects():
            time_progression = self.get_wait_time_progression(duration, stop_condition)
            # TODO, be smart about setting a static image
            # the same way Scene.play does
            last_t = 0
            for t in time_progression:
                dt = t - last_t
                last_t = t
                self.update_mobjects(dt)
                self.update_frame()
                self.add_frames(self.get_frame())
                if stop_condition is not None and stop_condition():
                    time_progression.close()
                    break
        elif file_writer_config["skip_animations"]:
            # Do nothing
            return self
        else:
            self.update_frame()
            dt = 1 / self.camera.frame_rate
            n_frames = int(duration / dt)
            frame = self.get_frame()
            self.add_frames(*[frame] * n_frames)
        return self

    def wait_until(self, stop_condition, max_time=60):
        """
        Like a wrapper for wait().
        You pass a function that determines whether to continue waiting,
        and a max wait time if that is never fulfilled.

        Parameters
        ----------
        stop_condition : function
            The function whose boolean return value determines whether to continue waiting

        max_time : int or float, optional
            The maximum wait time in seconds, if the stop_condition is never fulfilled.
        """
        self.wait(max_time, stop_condition=stop_condition)

    def force_skipping(self):
        """
        This forces the skipping of animations,
        by setting original_skipping_status to
        whatever skip_animations was, and setting
        skip_animations to True.

        Returns
        -------
        Scene
            The Scene, with skipping turned on.
        """
        self.original_skipping_status = file_writer_config["skip_animations"]
        file_writer_config["skip_animations"] = True
        return self

    def revert_to_original_skipping_status(self):
        """
        Forces the scene to go back to its original skipping status,
        by setting skip_animations to whatever it reads
        from original_skipping_status.

        Returns
        -------
        Scene
            The Scene, with the original skipping status.
        """
        if hasattr(self, "original_skipping_status"):
            file_writer_config["skip_animations"] = self.original_skipping_status
        return self

    def add_frames(self, *frames):
        """
        Adds a frame to the video_file_stream

        Parameters
        ----------
        *frames : numpy.ndarray
            The frames to add, as pixel arrays.
        """
        dt = 1 / self.camera.frame_rate
        self.increment_time(len(frames) * dt)
        if file_writer_config["skip_animations"]:
            return
        for frame in frames:
            self.file_writer.write_frame(frame)

    def add_sound(self, sound_file, time_offset=0, gain=None, **kwargs):
        """
        This method is used to add a sound to the animation.

        Parameters
        ----------
        sound_file : str
            The path to the sound file.

        time_offset : int,float, optional
            The offset in the sound file after which
            the sound can be played.

        gain :

        """
        if file_writer_config["skip_animations"]:
            return
        time = self.time + time_offset
        self.file_writer.add_sound(sound_file, time, gain, **kwargs)

    def show_frame(self):
        """
        Opens the current frame in the Default Image Viewer
        of your system.
        """
        self.update_frame(ignore_skipping=True)
        self.camera.get_image().show()


class EndSceneEarlyException(Exception):
    pass
