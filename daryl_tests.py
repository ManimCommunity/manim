from manim import *

# class Test(Scene):
# 	def construct(self):
# 		a = Sphere2()
# 		self.add(a)
# 		self.wait()

class ConeTest(ThreeDScene):
	def construct(self):
		b = Cone(show_base=True, fill_color=GREEN)
		self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
		# self.begin_ambient_camera_rotation(rate=1)
		self.add(b)
		# self.wait(3)

class ArrowTest(ThreeDScene):
	def construct(self):
		a = Arrow3D(start = np.array([0, 0, 0]), end = np.array([2, 2, 2]))
		self.add(a)
		self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
		print(a.start)
		# self.begin_ambient_camera_rotation(rate=1)
		# self.wait(5)

class CylinderTest(ThreeDScene):
	def construct(self):
		cyl = Cylinder()
		self.add(cyl)
		self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
		# self.begin_ambient_camera_rotation(rate=1)
		# self.wait(5)

class Line3DTest(ThreeDScene):
	def construct(self):
		line = Line3D(start = np.array([0, 0, 0]), end = np.array([2, 2, 2]))
		self.add(line)
		self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

class LineTest(Scene):
	def construct(self):
		line = Line(np.array([0, 0, 0]), np.array([2, 2, 0]))
		self.add(line)
		print(line.start)
