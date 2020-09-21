Shapes
=================================

.. manim:: Example1Shape
    :quality: medium
    :save_last_frame:

    class Shape1(Scene):
        def construct(self):
            d = Dot()
            c = Circle()
            s = Square()
            t = Triangle()
            d.next_to(c, RIGHT)
            s.next_to(c, LEFT)
            t.next_to(c, DOWN)
            self.add(d, c, s, t)
            self.wait(1)

.. manim:: Example1ImageFromArray
    :quality: medium
    :save_last_frame:

    class Example1ImageFromArray(Scene):
        def construct(self):
            image = ImageMobject(np.uint8([[0, 100, 30, 200],
                                           [255, 0, 5, 33]]))
            image.set_height(7)
            self.add(image)

.. manim:: Example1ImageFromArray
    :quality: medium
    :save_last_frame:

    class Example2ImageFromFile(Scene):
        def construct(self):
            # these four lines, when you want to import an image from the web
            import requests
            from PIL import Image
            img = Image.open(requests.get("https://raw.githubusercontent.com/ManimCommunity/manim/master/logo/cropped.png",
                                          stream=True).raw)
            img_mobject = ImageMobject(img)
            # this line, when you want to import your Image on your machine
            # img_mobject = ImageMobject("<your image address>")
            img_mobject.scale(3)
            self.add(img_mobject)