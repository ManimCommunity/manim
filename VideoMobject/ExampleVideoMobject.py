from manim import *

class VideoMobjectTest(Scene):
    def construct(self):
        file_directory = str(Path.home() / 'Downloads' / "VideoTest.mp4")
        from VideoMobject.VideoMobject import VideoMobject
        vid = VideoMobject(file_directory)
        self.add(vid.image_canvas)
        self.play(ShowCreation(SurroundingRectangle(vid.image_canvas)))
        self.wait()
        vid.play()
        self.wait(1)
        vid.pause()
        vid.image_canvas.scale(0.5)
        self.wait(1)
        vid.play()
        self.wait()

from pathlib import Path
if __name__ == "__main__":
    script = f"{Path(__file__).resolve()}"
    os.system(f"manim  -l -p -c 'BLACK' --video_dir ~/Downloads/ " + script )