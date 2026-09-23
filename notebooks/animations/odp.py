"""ODP concept animation; timings are illustrative, not benchmark measurements.

Preview (960 x 540, 30 fps; enough frames for the fast decode sweep):
    notebooks/animations/.venv/bin/manim -ql --fps 30 -r 960,540 \
        --media_dir notebooks/animations/media notebooks/animations/odp.py ODP

Only Manim's Text is used: no LaTeX installation is required.
"""

from pathlib import Path

import manimpango
from manim import (
    AnimationGroup, Arrow, Create, DashedLine, DOWN, FadeIn, FadeOut,
    LaggedStart, LEFT, Line, ManimColor, MovingCameraScene,
    PI, Rectangle, RIGHT, RoundedRectangle, ShowPassingFlash, Square, Text,
    Succession, UP, UpdateFromAlphaFunc, VGroup, Wait, config, interpolate_color, linear, smooth,
)


config.background_color = "#0B0F19"  # Hugging Face dark-theme background.
config.frame_width = 16
config.frame_height = 9

# Register only for this render process; no system-wide font installation.
FONT_DIR = Path(__file__).resolve().parents[2] / "01_BRAND_ASSETS/fonts"
for font_file in ("NVIDIASans_Rg.ttf", "NVIDIASans_Md.ttf", "NVIDIASans_Bd.ttf"):
    if not manimpango.register_font(str(FONT_DIR / font_file)):
        raise RuntimeError(f"Could not register animation font: {FONT_DIR / font_file}")

# NVIDIA Dark / Focus: neutral baseline, green takeaway, secondary colors for roles.
INK = "#FFFFFF"
MUTED = "#F7F7F7"
EDGE = "#757575"
PANEL = "#313131"
LOAD = "#EF9100"
COMPUTE = "#76B900"
DECODE = "#757575"
CACHE = "#1DBBA4"
PROMPT = CACHE

N_LAYERS = 6
PREFILL_SPEEDUP = 1.2
PREFILL_SECONDS = 1.8 / PREFILL_SPEEDUP
DECODE_SECONDS = 0.05
XS, XP, XC, XD = -6.70, -3.45, -0.1, 3.45
YS = [-1.90 + i * 0.70 for i in range(N_LAYERS)]


def text(value, size=20, color=INK, weight="NORMAL"):
    # Shape at a larger size before scaling the vector outlines: small Pango
    # sizes can round glyph advances enough to make word spacing uneven.
    return Text(value, font="NVIDIA Sans", weight=weight, font_size=64, color=color).scale(size / 64)


def wordmark():
    return (text("Nemotron Labs",64,weight="BOLD").rotate(PI/2)
            .scale_to_fit_height(7.5).move_to([7.48,0,0]).set_z_index(10))


class Layer(VGroup):
    """One logical layer: only Loading / Ready / Computing imply residency."""

    def __init__(self, index, x, resident=False):
        super().__init__()
        self.box = RoundedRectangle(
            width=2.40, height=0.52, corner_radius=0.07,
            stroke_color=DECODE if resident else EDGE, stroke_width=1.4,
            fill_color=DECODE if resident else PANEL,
            fill_opacity=0.20 if resident else 0.65,
        ).move_to([x, YS[index], 0])
        self.fill = Rectangle(width=0.001, height=0.46, stroke_width=0,
                              fill_color=LOAD, fill_opacity=0.65)
        self.fill.move_to(self.box).align_to(self.box, LEFT).shift(RIGHT * 0.03)
        self.label = text("Resident" if resident else "", 17,
                          INK if resident else MUTED).move_to(self.box)
        self.current_label = "Resident" if resident else ""
        self.offloaded_label = text("Offloaded",17,MUTED).move_to(self.box)
        self.offloaded_label.set_opacity(0 if resident else .5)
        self.add(self.box, self.fill, self.label, self.offloaded_label)
        if not resident:
            self.box.set_stroke(opacity=.5).set_fill(opacity=.325)

    def status(self, value, color=INK):
        if value != self.current_label:
            self.label.become(text(value, 17, color).move_to(self.box))
            self.current_label = value
        return self

    def progress(self, fraction, color):
        self.fill.set_fill(color, opacity=.58 if fraction else 0)
        self.fill.stretch_to_fit_width(max(.001, 2.34 * fraction))
        self.fill.move_to(self.box).align_to(self.box, LEFT).shift(RIGHT * .03)

    def bar(self, label, color, finish_early=False, release_on_finish=False):
        self.status(label)
        self.label.set_opacity(0)
        self.progress(0, color)
        def update(_, alpha):
            entrance = smooth(min(1, alpha * 8))
            self.offloaded_label.set_opacity(.5*(1-entrance) if label == "Loading" else 0)
            self.box.set_stroke(opacity=.5+.5*entrance).set_fill(opacity=.325+.325*entrance)
            f = min(1, alpha * 1.6) if finish_early else alpha
            self.progress(f, color)
            if finish_early and f >= 1:
                self.status("Ready", INK)
                self.label.set_opacity(smooth(min(1, (alpha-.625)*10)))
            else:
                self.label.set_opacity(entrance)
            if release_on_finish:
                fade = smooth(max(0,min(1,(alpha-.9)/.1)))
                self.fill.set_fill(opacity=.58*(1-fade))
                self.label.set_opacity(entrance*(1-fade))
                self.offloaded_label.set_opacity(.5*fade)
                self.box.set_stroke(opacity=1-.5*fade).set_fill(opacity=.65-.325*fade)
        return UpdateFromAlphaFunc(self, update, rate_func=linear)


class ODP(MovingCameraScene):
    background_color = "#0B0F19"

    def construct(self):
        self.camera.background_color = self.background_color
        title = text("Offloaded Disaggregated Prefill (ODP)", 32, weight="MEDIUM").move_to([0, 3.94, 0])
        self.branding = wordmark()

        # Persistent storage at left; all other columns live on the device.
        device = RoundedRectangle(width=11.3, height=5.64, corner_radius=.15,
                                  stroke_color=EDGE, stroke_width=1.2,
                                  fill_opacity=0).move_to([.03, .18, 0])
        device_label = text("Device Memory", 16, MUTED, weight="MEDIUM").move_to([.03, 2.78, 0])
        columns = VGroup(
            text("SSD", 24, LOAD, weight="MEDIUM").move_to([XS, 2.43, 0]),
            text("Prefiller", 24, weight="MEDIUM").move_to([XP, 2.43, 0]),
            text("KV Cache", 24, CACHE, weight="MEDIUM").move_to([XC, 2.43, 0]),
            text("Decode Model", 24, INK, weight="MEDIUM").move_to([XD, 2.43, 0]),
            text("Prefiller storage", 15, MUTED).move_to([XS, 2.07, 0]),
            text("NVFP4, offloaded", 15, MUTED).move_to([XP, 2.07, 0]),
            text("Weight-only, always on device", 15, MUTED).move_to([XD, 2.07, 0]),
        )
        prefill = [Layer(i, XP) for i in range(N_LAYERS)]
        decode = [Layer(i, XD, True) for i in range(N_LAYERS)]
        ssd, cache_boxes, cache_fills, load_paths, write_paths, read_paths = [], [], [], [], [], []
        for i, y in enumerate(YS):
            disk = RoundedRectangle(width=1.32, height=.46, corner_radius=.05,
                                    stroke_color=LOAD, stroke_width=1,
                                    fill_color=LOAD, fill_opacity=.1).move_to([XS, y, 0])
            ssd.append(VGroup(disk, text(f"Layer {i+1}", 17, LOAD).move_to(disk)))
            cache = RoundedRectangle(width=1.62, height=.46, corner_radius=.05,
                                     stroke_color=EDGE, stroke_width=1,
                                     fill_color=PANEL, fill_opacity=.6).move_to([XC, y, 0])
            cache_boxes.append(VGroup(cache, text("KV", 16, MUTED).move_to(cache)))
            # Leave room at right for the newly generated tokens' KV entries.
            fill = Rectangle(width=.001, height=.40, stroke_width=0,
                             fill_color=CACHE, fill_opacity=.3).move_to(cache).align_to(cache, LEFT).shift(RIGHT*.03)
            cache_fills.append(fill)
            load_paths.append(DashedLine(disk.get_right(), prefill[i].box.get_left(),
                                         dash_length=.08, color=EDGE, stroke_width=1))
            write_paths.append(Arrow(prefill[i].box.get_right(), cache.get_left(),
                                     buff=.10, color=EDGE, stroke_width=1.3,
                                     max_tip_length_to_length_ratio=.13))
            read_paths.append(Arrow(cache.get_right(), decode[i].box.get_left(),
                                    buff=.10, color=EDGE, stroke_width=1.3,
                                    max_tip_length_to_length_ratio=.13))
        up_prefill, up_decode = [], []
        for column, arrows in ((prefill, up_prefill), (decode, up_decode)):
            for i in range(N_LAYERS - 1):
                arrows.append(Arrow(column[i].box.get_top(), column[i+1].box.get_bottom(),
                                    buff=.015, color=MUTED, stroke_width=2.0,
                                    max_tip_length_to_length_ratio=.4))

        prompt_box = RoundedRectangle(width=4.2, height=.74, corner_radius=.10,
                                      stroke_color=PROMPT, stroke_width=1.3,
                                      fill_color=PANEL, fill_opacity=1).move_to([XP, -3.51, 0])
        prompt_label = text("Prompt", 14, MUTED, weight="MEDIUM").move_to([XP-1.65, -2.95, 0])
        prompt = text("Why do leaves look green?", 20, PROMPT).move_to(prompt_box)
        output_box = RoundedRectangle(width=5.2, height=1.0, corner_radius=.10,
                                      stroke_color=DECODE, stroke_width=1.3,
                                      fill_color=PANEL, fill_opacity=1).move_to([XD, -3.55, 0])
        output_label = text("Output", 14, MUTED, weight="MEDIUM").move_to([XD-2.10, -2.95, 0])
        input_arrow = Arrow(prompt_box.get_top(), prefill[0].box.get_bottom(),
                            buff=.10, color=PROMPT, stroke_width=2)
        self.play(FadeIn(title),FadeIn(self.branding),Create(device),FadeIn(device_label),
                  FadeIn(columns),run_time=.5)
        self.play(
            LaggedStart(*[FadeIn(VGroup(ssd[i],prefill[i],cache_boxes[i],decode[i]))
                          for i in range(N_LAYERS)],lag_ratio=.08),
            Succession(Wait(.15),AnimationGroup(
                *[Create(path) for path in [*load_paths,*write_paths,*read_paths,
                                           *up_prefill,*up_decode]],run_time=.5)),
            FadeIn(VGroup(prompt_box,prompt_label,prompt,output_box,output_label)),
            Create(input_arrow),run_time=.75)
        self.wait(.4)
        self.play(prefill[0].bar("Loading", LOAD),
                  ShowPassingFlash(load_paths[0].copy().set_color(LOAD), time_width=.7),
                  run_time=1.1 / PREFILL_SPEEDUP)

        packet = VGroup(*[Square(side_length=.13, stroke_width=0, fill_color=PROMPT,
                                  fill_opacity=1) for _ in range(4)]).arrange(RIGHT, buff=.035)
        packet.move_to(prompt_box.get_top()+UP*.18)
        self.play(FadeIn(packet), run_time=.2)
        for i in range(N_LAYERS):
            # Handoff animates inside the compute interval, never
            # between intervals: layer i+1 begins as soon as layer i finishes.
            animations = [
                prefill[i].bar("Computing", COMPUTE, release_on_finish=True),
                packet.animate(rate_func=lambda a: smooth(min(1, a*8))).move_to(
                    prefill[i].box.get_bottom()+DOWN*.10),
                Succession(
                    Wait(PREFILL_SECONDS-.22),
                    ShowPassingFlash(write_paths[i].copy().set_color(CACHE),
                                     time_width=.65, run_time=.22),
                ),
            ]
            # Form each cache smoothly during the end of its layer's compute,
            # without adding a pause before the next layer starts.
            cache_fills[i].set_fill(opacity=0)
            self.add(cache_fills[i])
            def write_cache(_, alpha, i=i):
                fraction = smooth(max(0, min(1, (alpha-.85)/.15)))
                color = interpolate_color(ManimColor(EDGE),ManimColor(CACHE),fraction)
                cache_boxes[i][0].set_stroke(color)
                cache_boxes[i][1].set_color(interpolate_color(ManimColor(MUTED),ManimColor(CACHE),fraction))
                cache_fills[i].set_fill(opacity=.3*fraction)
                cache_fills[i].stretch_to_fit_width(max(.001,1.03*fraction)).align_to(
                    cache_boxes[i][0],LEFT).shift(RIGHT*.03)
                if i<N_LAYERS-1:
                    up_prefill[i].set_color(interpolate_color(ManimColor(MUTED),ManimColor(PROMPT),fraction))
            animations.append(UpdateFromAlphaFunc(
                VGroup(cache_boxes[i],cache_fills[i],*up_prefill[i:i+1]),
                write_cache,rate_func=linear))
            if i+1<N_LAYERS:
                animations += [prefill[i+1].bar("Loading", LOAD, finish_early=True),
                               ShowPassingFlash(load_paths[i+1].copy().set_color(LOAD), time_width=.65)]
            self.play(*animations, run_time=PREFILL_SECONDS)
        self.play(FadeOut(packet), FadeOut(input_arrow),run_time=.25)

        # The final prefill hidden state gives the first token; decode starts from it.
        # Stable word chips: previously emitted words never morph or get redrawn.
        words = ["Leaves", "look", "green", "because", "of", "chlorophyll."]
        chips = []
        x_left = output_box.get_left()[0] + .15
        cursor, row_y = x_left, output_box.get_top()[1]-.27
        for word in words:
            label = text(word, 18)
            chip = RoundedRectangle(width=label.width+.20, height=.36,
                                    corner_radius=.055, stroke_width=1,
                                    stroke_color=DECODE, fill_color=DECODE, fill_opacity=.13)
            if cursor + chip.width > output_box.get_right()[0] - .15:
                cursor, row_y = x_left, row_y - .44
            chip.move_to([cursor+chip.width/2, row_y, 0])
            label.move_to(chip)
            chips.append(VGroup(chip, label))
            cursor += chip.width + .065
        self.play(FadeIn(chips[0], shift=UP*.10, scale=.9), run_time=.30)
        self.wait(.65)
        feedback = Arrow(output_box.get_top(), decode[0].box.get_bottom(),
                         buff=.10, color=CACHE, stroke_width=2)
        return_x = device.get_right()[0]-.30
        result_path = VGroup(
            Line(decode[-1].box.get_right(), [return_x, YS[-1], 0], color=CACHE, stroke_width=1.5),
            # Down: the last layer produces an output token. The separate upward
            # feedback arrow feeds that token into the next bottom-to-top pass.
            Arrow([return_x, YS[-1], 0], [return_x, output_box.get_top()[1]+.04, 0], color=CACHE,
                  buff=0, stroke_width=2, tip_length=.15),
        )
        self.play(Create(feedback),Create(result_path,lag_ratio=1),run_time=.3)
        # One continuous pass avoids rounding each 50-ms layer up to a whole frame.
        # Include every changing object so Cairo does not cache arrows/cache fills
        # as a static background while the active layer moves up the stack.
        decode_stack = VGroup(*decode, *read_paths, *cache_fills, *up_decode)
        for token in range(1,len(words)):
            def sweep(_, alpha, token=token):
                position = alpha * N_LAYERS
                for i, layer in enumerate(decode):
                    active = i <= position < i+1
                    layer.status("Computing" if active else "Resident", INK)
                    layer.progress(max(.15, position-i) if active else 0, COMPUTE)
                    layer.box.set_stroke(COMPUTE if active else DECODE)
                    read_paths[i].set_color(CACHE if active else EDGE)
                    fraction = min(1, max(0, position-i))
                    cache_fills[i].stretch_to_fit_width(1.03+.075*(token-1+fraction)).align_to(cache_boxes[i][0],LEFT).shift(RIGHT*.03)
                    if i<N_LAYERS-1:
                        up_decode[i].set_color(CACHE if position>=i+1 else MUTED)
            self.play(UpdateFromAlphaFunc(decode_stack,sweep,rate_func=linear),
                      run_time=N_LAYERS*DECODE_SECONDS)
            self.play(FadeIn(chips[token],shift=UP*.10,scale=.9),
                      ShowPassingFlash(result_path.copy().set_color(CACHE),time_width=.6),
                      run_time=.10)
        self.play(FadeOut(feedback),run_time=.35)
        self.wait(1.8)
