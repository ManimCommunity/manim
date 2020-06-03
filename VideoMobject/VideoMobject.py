from pathlib import Path
from manim import *



class VideoMobject:
    def __init__(self, folder_name: str):
        self.frames= self.init_frames(folder_name)
        self.current_frame_num = 0
        self.image_canvas = self.get_image(self.current_frame_num)
        self.frames_length= len(self.frames)

    def get_image(self,frame_num):
        image_path = self.frames[frame_num]
        image = Image.open(str(image_path))
        return ImageMobject(np.array(image)).scale(3)

    def init_frames(self, file_directory):
        file_directory = Path(file_directory)
        name_stem = file_directory.stem
        name_ext = file_directory.suffix
        parent_directory = file_directory.parent
        frame_folder_name = name_stem
        frames_directory = parent_directory / frame_folder_name
        if not frames_directory.is_dir():
            print("File does not exist")
            frames_directory.mkdir(parents=True, exist_ok=False)
            print(frames_directory)
            # first one is the Downloads/video.mp4, second one is Downloads/video/video_001.png
            command = f"ffmpeg -i '{frames_directory}{name_ext}'  {frames_directory / frame_folder_name}_%03d.png"
            os.system(command)
        else:
            print("File exist")

        frames = []
        for filename in sorted(frames_directory.glob('*.png')):
            frames.append(filename)
        print(len(frames))
        return frames

    def to_fist_frame(self):
        self.image_canvas.become(self.get_image(0))
    def to_last_frame(self):
        self.image_canvas.become(self.get_image(self.frames_length - 1))

    def play(self):
        self.image_canvas.add_updater(self.frame_updater)

    def pause(self):
        self.image_canvas.remove_updater(self.frame_updater)

    def frame_updater(self, mob, dt):
        self.current_frame_num += 1
        if (self.frames_length > self.current_frame_num):
            self.image_canvas.become(self.get_image(self.current_frame_num))