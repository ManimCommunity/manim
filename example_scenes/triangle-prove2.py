import numpy as np

from manim import *


class TriangleParallelProof(Scene):
    def construct(self):
        # 设置背景为黑色
        self.camera.background_color = BLACK

        # 创建一个三角形
        triangle = Triangle(color=WHITE, stroke_width=2)
        triangle.scale(2)

        # 获取顶点
        vertices = triangle.get_vertices()
        A, B, C = vertices

        # 为顶点添加标签
        labels = []
        for i, point in enumerate(vertices):
            label = Text(["A", "B", "C"][i], font_size=36).next_to(
                point, point - triangle.get_center()
            )
            labels.append(label)

        # 创建三个内角
        angle_radius = 0.4
        angle_A = Angle(
            Line(A, B), Line(A, C), radius=angle_radius, color=YELLOW, stroke_width=8
        )

        angle_B = Angle(
            Line(B, C), Line(B, A), radius=angle_radius, color=RED, stroke_width=8
        )

        angle_C = Angle(
            Line(C, A), Line(C, B), radius=angle_radius, color=BLUE, stroke_width=8
        )

        # 显示三角形和标签
        self.play(Create(triangle))
        self.play(*[Write(label) for label in labels])
        self.wait(0.5)

        # 显示角度
        angles = [angle_A, angle_B, angle_C]
        for angle in angles:
            self.play(Create(angle))
        self.wait()

        # 创建通过A点且平行于BC的直线
        # 计算BC的方向向量
        bc_vector = C - B
        # 计算平行线的起点和终点
        parallel_start = A + LEFT * 2
        parallel_end = A + RIGHT * 2

        # 创建平行线
        parallel_line = Line(parallel_start, parallel_end, color=WHITE).move_to(A)

        # 旋转平行线使其平行于BC
        angle_to_rotate = np.arctan2(bc_vector[1], bc_vector[0])
        parallel_line.rotate(angle_to_rotate, about_point=A)

        # 显示平行线
        self.play(Create(parallel_line))

        # 调整平行线标注的位置
        parallel_text = Text("平行于BC", font_size=24).next_to(
            parallel_line, UP, buff=0.85
        )  # 增加 buff 值
        self.play(Write(parallel_text))
        self.wait()

        # 创建两个内错角（分别与角B和角C相等）
        # 与角B对应的内错角
        parallel_angle_B = Angle(
            Line(A, parallel_start),
            Line(A, B),
            radius=angle_radius,
            color=RED,
            stroke_width=8,
        )

        # 与角C对应的内错角
        parallel_angle_C = Angle(
            Line(A, parallel_end),
            Line(A, C),
            radius=angle_radius,
            color=BLUE,
            stroke_width=8,
            other_angle=True,
        )

        # 显示内错角
        self.play(Create(parallel_angle_B))
        self.play(Create(parallel_angle_C))

        # 添加内错角相等的说明
        equal_angles_text_B = Text("内错角相等", font_size=24)
        equal_angles_text_B.next_to(parallel_angle_B, LEFT)

        equal_angles_text_C = Text("内错角相等", font_size=24)
        equal_angles_text_C.next_to(parallel_angle_C, RIGHT)

        self.play(Write(equal_angles_text_B))
        self.play(Write(equal_angles_text_C))
        self.wait()

        # 直接添加180度标注
        degree_text = MathTex(r"180^\circ", font_size=36)
        degree_text.next_to(A, UP * 0.35)  # 调整位置使其靠近A点
        self.play(Write(degree_text))

        # 添加最终结论
        conclusion = Text("所以三角形的三个内角和为180°", font_size=36)
        conclusion.to_edge(UP)
        self.play(Write(conclusion))

        self.wait(2)


if __name__ == "__main__":
    with tempconfig({"quality": "medium_quality", "preview": True}):
        scene = TriangleParallelProof()
        scene.render()
