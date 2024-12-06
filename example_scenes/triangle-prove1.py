from manim import *
import numpy as np

# 定义一个名为TriangleAnglesSum的类，它继承自Scene类，用于构建特定的动画场景
class TriangleAnglesSum(Scene):
    def construct(self):
        # 设置场景相机的背景颜色为黑色，这样整个动画场景的背景就是黑色的了
        self.camera.background_color = BLACK

        # 创建一个白色边框、边框宽度为2的三角形，并将其大小缩放为原来的2倍
        triangle = Triangle(color=WHITE, stroke_width=2)
        triangle.scale(2)

        # 获取三角形的顶点坐标，返回的是一个包含顶点坐标的数组
        vertices = triangle.get_vertices()
        # 将顶点坐标数组中的元素分别赋值给A、B、C，方便后续使用
        A, B, C = vertices

        # 创建一个空列表，用于存放后面要添加到场景中的顶点标签
        labels = []
        # 遍历顶点坐标数组以及对应的索引
        for i, point in enumerate(vertices):
            # 根据索引创建对应的顶点标签文本，字体大小为36
            # 通过计算使标签位于对应顶点且相对三角形中心有一定偏移的位置
            label = Text(["A", "B", "C"][i], font_size=36).next_to(point, point - triangle.get_center())
            labels.append(label)

        # 设置角度相关的一些参数（这里角度用扇形来大致表示，设置扇形半径等属性）
        angle_radius = 0.4  # 扇形半径，用于控制表示角的扇形大小
        angle_arc_angle = PI / 3  # 扇形的圆心角大小，这里设置为60度（PI/3弧度），可根据实际情况调整
        angle_stroke_width = 8  # 扇形边框宽度

        # 创建表示角A的扇形对象，指定扇形的圆心、两条半径对应的点以及相关属性
        angle_A = Sector(
            start_angle=0,
            angle=angle_arc_angle,
            radius=angle_radius,
            stroke_color=YELLOW,
            fill_color=YELLOW,
            stroke_width=angle_stroke_width,
            fill_opacity=0.3  # 设置一定透明度，让效果更美观
        ).move_arc_center_to(A).rotate(4*PI/3,about_point=A)

        # 创建表示角B的扇形对象，指定扇形的圆心、两条半径对应的点以及相关属性
        angle_B = Sector(
            start_angle=0,
            angle=angle_arc_angle,
            radius=angle_radius,
            stroke_color=RED,
            fill_color=RED,
            stroke_width=angle_stroke_width,
            fill_opacity=0.3
        ).move_arc_center_to(B)

        # 创建表示角C的扇形对象，指定扇形的圆心、两条半径对应的点以及相关属性
        angle_C = Sector(
            start_angle=0,
            angle=angle_arc_angle,
            radius=angle_radius,
            stroke_color=BLUE,
            fill_color=BLUE,
            stroke_width=angle_stroke_width,
            fill_opacity=0.3
        ).move_arc_center_to(C).rotate(5*PI/3 + PI,about_point=C)

        # 将创建好的三个表示角的扇形对象放入一个列表中，方便后续统一操作
        angles = [angle_A, angle_B, angle_C]

        # 1. 显示三角形和顶点标签
        # 播放创建三角形的动画，将三角形添加到场景中
        self.play(Create(triangle))
        # 依次播放写入每个顶点标签的动画，将标签添加到场景中
        self.play(*[Write(label) for label in labels])
        # 等待0.5秒，让观众能看清当前画面
        self.wait(0.5)

        # 2. 显示角度（这里是显示用扇形表示的角）
        for angle in angles:
            self.play(Create(angle))
        self.wait()

        # 创建一个空列表，用于存放后续复制并移动后的角度扇形副本
        angle_copies = []

        # 设置底部参考线的中心点坐标，通过将三角形中心向下移动一定距离来确定
        bottom_line_center = triangle.get_center() + DOWN * 3

        # 计算角度A移动后的目标位置坐标，就是底部参考线的中心点位置
        target_point_A = bottom_line_center
        # 计算角度B移动后的目标位置坐标，在底部参考线中心点基础上向左、向上有一定偏移
        target_point_B = bottom_line_center + LEFT * 0.53 + UP * 0.09
        # 计算角度C移动后的目标位置坐标，在底部参考线中心点基础上向右、向上有一定偏移
        target_point_C = bottom_line_center + RIGHT * 0.53 + UP * 0.09

        # 移动和旋转A角（这里的角用扇形表示）
        # 复制角度A对应的扇形得到一个副本
        copy_angle_A = angle_A.copy()
        angle_copies.append(copy_angle_A)
        # 播放动画，将复制的角度A对应的扇形移动到目标位置，动画运行时间为1.5秒
        self.play(
            copy_angle_A.animate.move_to(target_point_A),
            run_time=1.5
        )
        # 等待0.3秒，让观众看清移动后的效果
        self.wait(0.3)

        # 移动和旋转B角（这里的角用扇形表示）
        # 复制角度B对应的扇形得到一个副本
        copy_angle_B = angle_B.copy()
        angle_copies.append(copy_angle_B)
        # 播放动画，将复制的角度B对应的扇形移动到目标位置并旋转PI弧度（即180度），动画运行时间为1.5秒
        self.play(
            copy_angle_B.animate.move_to(target_point_B).rotate(PI),
            run_time=1.5
        )
        # 等待0.3秒，让观众看清移动和旋转后的效果
        self.wait(0.3)

        # 移动和旋转C角（这里的角用扇形表示）
        # 复制角度C对应的扇形得到一个副本
        copy_angle_C = angle_C.copy()
        angle_copies.append(copy_angle_C)
        # 播放动画，将复制的角度C对应的扇形移动到目标位置并旋转PI弧度（即180度），动画运行时间为1.5秒
        self.play(
            copy_angle_C.animate.move_to(target_point_C).rotate(PI),
            run_time=1.5
        )
        # 等待0.3秒，让观众看清移动和旋转后的效果
        self.wait(0.3)

        # 创建底部的直线（表示180度）
        # 计算直线的总宽度，基于之前设置的角度扇形半径来确定
        total_width = angle_radius * 4
        # 创建一条白色的直线，起点和终点坐标根据底部参考线中心点以及计算出的左右偏移量来确定，并且有向上的一定偏移
        line = Line(
            bottom_line_center + LEFT * (total_width / 2) + UP * 0.52,
            bottom_line_center + RIGHT * (total_width / 2) + UP * 0.52,
            color=WHITE
        )
        # 播放创建直线的动画，将直线添加到场景中
        self.play(Create(line))

        # 添加180度标注
        # 创建一个表示“180°”的数学文本对象，字体大小为48，并将其放置在直线下方一定距离处
        degree_text = MathTex(r"180^\circ", font_size=48).next_to(line, DOWN * 5.2)
        # 播放写入标注文本的动画，将标注添加到场景中
        self.play(Write(degree_text))

        # 添加闪烁效果
        # 对每个角度扇形副本添加闪烁指示效果，通过缩放来实现闪烁，动画运行时间为1秒
        self.play(
            *[Indicate(copy, scale_factor=1.2) for copy in angle_copies],
            run_time=1
        )

        # 添加总结文字
        # 创建一个文本对象，表示“三角形内角和为180°”，字体大小为36，并将其放置在场景上方边缘处
        conclusion = Text("三角形内角和为180°", font_size=36)
        conclusion.to_edge(UP)
        # 播放写入总结文字的动画，将总结文字添加到场景中
        self.play(Write(conclusion))

        # 等待2秒，让观众有足够时间看清最后的总结画面
        self.wait(2)


if __name__ == "__main__":
    # 使用临时配置，设置动画渲染的质量为中等质量，并开启预览模式
    with tempconfig({"quality": "medium_quality", "preview": True}):
        # 创建TriangleAnglesSum类的实例，即创建一个具体的动画场景对象
        scene = TriangleAnglesSum()
        # 调用render方法来渲染动画场景，生成对应的动画视频等输出内容
        scene.render()