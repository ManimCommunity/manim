from manim import *


class ConversationFlowScene(ThreeDScene):
    """Animates the flow of a conversation between a user and a chatbot named Kate,
    then reveals metadata and applies evaluation rules one by one.
    """

    def construct(self):
        # Camera setup for 3D perspective
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)

        # ---- Build conversation messages ----
        messages = [
            ("User", "Hi Kate, what's the weather like today?", BLUE_C),
            (
                "Kate",
                "The forecast shows partly cloudy skies\nwith a high of 72°F.",
                GREEN_C,
            ),
            ("User", "Any chance of rain this afternoon?", BLUE_C),
            (
                "Kate",
                "There's a 30% chance of showers\nafter 3 PM in your area.",
                GREEN_C,
            ),
        ]

        message_mobjects = []
        for sender, text, color in messages:
            bubble = self._create_message_bubble(sender, text, color)
            message_mobjects.append(bubble)

        conversation_group = VGroup(*message_mobjects)
        conversation_group.arrange(DOWN, buff=0.35)
        conversation_group.move_to(ORIGIN + UP * 1.5)

        # Fix all 2D elements in frame so they don't rotate with camera
        self.add_fixed_in_frame_mobjects(*message_mobjects)

        # ---- Animate messages popping up one by one ----
        for i, msg in enumerate(message_mobjects):
            self.play(FadeIn(msg, shift=UP * 0.3), run_time=0.8)
            self.wait(0.4)

        self.wait(0.5)

        # ---- Build metadata section ----
        metadata_items = [
            "Presented Flows",
            "System Prompt",
            "FAQs",
            "MyNWS Articles",
            "Tool Calls",
            "Collected Variables",
            "Metadata",
        ]

        metadata_title = Text("── Context & Metadata ──", font_size=20, color=YELLOW)
        metadata_entries = VGroup()
        for item in metadata_items:
            entry = self._create_metadata_entry(item)
            metadata_entries.add(entry)

        metadata_entries.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        metadata_section = VGroup(metadata_title, metadata_entries)
        metadata_section.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        metadata_section.next_to(conversation_group, DOWN, buff=0.5)

        self.add_fixed_in_frame_mobjects(metadata_title, *metadata_entries)

        # Animate metadata appearing
        self.play(FadeIn(metadata_title, shift=UP * 0.2), run_time=0.6)
        self.play(
            LaggedStart(
                *(FadeIn(entry, shift=UP * 0.15) for entry in metadata_entries),
                lag_ratio=0.2,
            ),
            run_time=2.0,
        )
        self.wait(0.5)

        # ---- Shift everything left to make room for rules ----
        all_content = VGroup(conversation_group, metadata_section)
        self.play(all_content.animate.shift(LEFT * 2.5), run_time=0.8)
        self.wait(0.3)

        # ---- Build evaluation rules on the right ----
        rules = [
            "1. Tone must be professional and helpful",
            "2. Responses must be factually accurate",
            "3. No hallucinated data or sources",
            "4. Must reference local forecast data",
            "5. Greetings should use the bot name",
            "6. Tool calls must match user intent",
            "7. Metadata fields must be populated",
        ]

        rule_mobjects = []
        for rule_text in rules:
            rule = Text(rule_text, font_size=18, color=WHITE)
            rule_mobjects.append(rule)

        rules_group = VGroup(*rule_mobjects)
        rules_group.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        rules_group.to_edge(RIGHT, buff=0.5).shift(UP * 0.5)

        self.add_fixed_in_frame_mobjects(*rule_mobjects)

        # Animate rules sliding in from the right
        for rule_mob in rule_mobjects:
            rule_mob.shift(RIGHT * 4)  # Start off-screen

        for rule_mob in rule_mobjects:
            self.play(rule_mob.animate.shift(LEFT * 4), run_time=0.5)
            self.wait(0.15)

        self.wait(0.5)

        # ---- Apply rules one by one with highlights and connections ----
        # Map each rule to a target (message or metadata entry)
        targets = [
            message_mobjects[1],  # Rule 1 -> Kate's first response (tone)
            message_mobjects[3],  # Rule 2 -> Kate's second response (accuracy)
            message_mobjects[3],  # Rule 3 -> Kate's second response (no hallucination)
            metadata_entries[0],  # Rule 4 -> Presented Flows (local forecast)
            message_mobjects[1],  # Rule 5 -> Kate's first response (greeting/name)
            metadata_entries[4],  # Rule 6 -> Tool Calls
            metadata_entries[6],  # Rule 7 -> Metadata
        ]

        for rule_mob, target in zip(rule_mobjects, targets):
            # Highlight the rule
            highlight_box = SurroundingRectangle(
                rule_mob, color=YELLOW, buff=0.08, stroke_width=2
            )
            self.add_fixed_in_frame_mobjects(highlight_box)

            # Create connecting arrow from rule to target
            arrow = Arrow(
                rule_mob.get_left(),
                target.get_right(),
                color=YELLOW,
                stroke_width=2,
                buff=0.1,
                max_tip_length_to_length_ratio=0.15,
            )
            self.add_fixed_in_frame_mobjects(arrow)

            # Highlight the target
            target_highlight = SurroundingRectangle(
                target, color=ORANGE, buff=0.08, stroke_width=2
            )
            self.add_fixed_in_frame_mobjects(target_highlight)

            # Animate: highlight rule, grow arrow, highlight target
            self.play(
                Create(highlight_box),
                GrowArrow(arrow),
                Create(target_highlight),
                run_time=0.8,
            )
            self.wait(0.6)

            # Fade out connection artifacts before next rule
            self.play(
                FadeOut(highlight_box),
                FadeOut(arrow),
                FadeOut(target_highlight),
                run_time=0.4,
            )

        self.wait(1.0)

    def _create_message_bubble(self, sender: str, text: str, color) -> VGroup:
        """Create a chat-bubble-style message with sender label."""
        label = Text(sender, font_size=16, color=color, weight=BOLD)
        body = Text(text, font_size=20, color=WHITE)

        content = VGroup(label, body)
        content.arrange(DOWN, buff=0.1, aligned_edge=LEFT)

        bubble = RoundedRectangle(
            corner_radius=0.15,
            width=content.width + 0.5,
            height=content.height + 0.3,
            color=color,
            fill_color=color,
            fill_opacity=0.12,
            stroke_width=1.5,
        )
        bubble.move_to(content)

        group = VGroup(bubble, content)
        return group

    def _create_metadata_entry(self, label: str) -> VGroup:
        """Create a metadata row with a bullet and label."""
        bullet = Dot(radius=0.04, color=YELLOW)
        text = Text(label, font_size=18, color=GREY_A)
        entry = VGroup(bullet, text)
        entry.arrange(RIGHT, buff=0.15)
        return entry
