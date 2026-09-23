"""Animate the exact pareto_gsq_rco_both data in the ODP video's visual style.

Results only:
    notebooks/animations/.venv/bin/manim -ql --fps 30 -r 960,540 \
        --media_dir notebooks/animations/media notebooks/animations/odp_results.py ODPResults
Full video (original mechanism followed by measured results): use ODPWithResults.
"""

import math

import numpy as np
from manim import (
    AnimationGroup, Arrow, Create, DashedLine, Dot, DOWN, FadeIn, FadeOut,
    LaggedStart, LEFT, Line, MovingCameraScene, RIGHT, Succession, Transform,
    UP, VGroup, VMobject, Wait, smooth,
)

from odp import CACHE, COMPUTE, DECODE, EDGE, INK, MUTED, ODP, text, wordmark
from pareto_data import load_results


class Chart(VGroup):
    """Explicit coordinates keep log axes and labels independent of LaTeX."""

    def __init__(self, left, x_range, y_range, x_ticks, y_ticks, xlabel,
                 log_y=False):
        super().__init__()
        self.left, self.bottom = left, -2.0
        self.width_units, self.height_units = 3.8, 3.55
        self.x_range, self.y_range, self.log_y = x_range, y_range, log_y
        self.add(Line([left, self.bottom, 0], [left, self.bottom+self.height_units, 0],
                      color=EDGE, stroke_width=1.2),
                 Line([left, self.bottom, 0], [left+self.width_units, self.bottom, 0],
                      color=EDGE, stroke_width=1.2))
        for value, label in x_ticks:
            point = self.point(value, y_range[0])
            self.add(Line(point, point+UP*self.height_units, color=EDGE,
                          stroke_width=.7, stroke_opacity=.5),
                     text(label, 16, MUTED).move_to(point+DOWN*.25))
        for value, label in y_ticks:
            point = self.point(x_range[0], value)
            self.add(Line(point, point+RIGHT*self.width_units, color=EDGE,
                          stroke_width=.7, stroke_opacity=.5),
                     text(label, 16, MUTED).next_to(point, LEFT, buff=.13))
        axis_label = text(xlabel,18,MUTED)
        self.add(axis_label.move_to([left+self.width_units/2, -2.66, 0]))

    def point(self, x, y):
        x0, x1 = map(math.log, self.x_range)
        fy = math.log if self.log_y else lambda value: value
        y0, y1 = map(fy, self.y_range)
        return np.array([self.left+(math.log(x)-x0)/(x1-x0)*self.width_units,
                         self.bottom+(fy(y)-y0)/(y1-y0)*self.height_units, 0])

    def curve(self, xs, ys, color, opacity=1):
        points = [self.point(x,y) for x,y in zip(xs,ys)]
        line = VMobject().set_points_as_corners(points).set_stroke(color, width=2.7, opacity=opacity)
        dots = VGroup(*[Dot(p,radius=.055,color=color,fill_opacity=opacity) for p in points])
        return VGroup(line,dots)

    def reference(self, y, label, color=MUTED, below=False):
        a,b = self.point(self.x_range[0],y), self.point(self.x_range[1],y)
        line = DashedLine(a,b,dash_length=.065,color=EDGE if color == MUTED else color,stroke_width=1.2)
        caption = text(label,15,color)
        caption.move_to(a + RIGHT*(caption.width/2+.06) + (DOWN if below else UP)*.18)
        return VGroup(line,caption)


def animate_results(scene):
    data = load_results()
    branding_entrance = []
    if not hasattr(scene,"branding"):
        scene.branding = wordmark()
        branding_entrance.append(FadeIn(scene.branding))
    title = text("Qwen3.8-27B: Adding an NVFP4 Prefiller",32,weight="MEDIUM").move_to([0,3.94,0])
    legend = VGroup(
        VGroup(Line(LEFT*.18,RIGHT*.18,color=DECODE,stroke_width=3),
               text("Weight-only",19,MUTED)).arrange(RIGHT,buff=.13),
        VGroup(Line(LEFT*.18,RIGHT*.18,color=COMPUTE,stroke_width=3),
               text("+ trained NVFP4 prefiller / ODP",19,COMPUTE)).arrange(RIGHT,buff=.13),
    ).arrange(RIGHT,buff=.65).move_to([0,3.20,0])
    panels = []
    for left, name, caption in [(-7.05,"MMLU-Pro","Text reasoning"),
                                (-1.95,"MMMU-Pro","Visual reasoning"),
                                (3.20,"Time to First Token","llama.cpp · DGX Spark")]:
        center=left+1.9
        panels += [text(name,25,weight="MEDIUM").move_to([center,2.45,0]),
                   text(caption,17,MUTED).move_to([center,2.03,0])]
    quality_axes=[Chart(left,(6,11.5),(20,92),[(6,"6"),(8,"8"),(10,"10")],
                        [(20,"20"),(40,"40"),(60,"60"),(80,"80")],"Device weights, GB")
                  for left in (-7.05,-1.95)]
    latency_axes=Chart(3.20,(900,36500),(.85,80),
                       [(n,f"{n//1024}K") for n in data["lengths"]],
                       [(1,"1s"),(10,"10s"),(60,"60s")],"Context length",log_y=True)
    ylabel=text("Accuracy, %",16,MUTED).rotate(math.pi/2).move_to([-7.79,-.25,0])
    scene.play(FadeIn(title),FadeIn(legend),*branding_entrance,*[FadeIn(p) for p in panels],
               *[FadeIn(a) for a in quality_axes],FadeIn(latency_axes),FadeIn(ylabel),run_time=.7)

    quality_curves, quality_targets, quality_ghosts = [], [], []
    bf16_references, format_labels = [], []
    for axes,bench in zip(quality_axes,("mmlu_pro","mmmu")):
        values=data["quality"][bench]
        baseline=axes.curve(data["sizes"],values["baseline"],DECODE)
        quality_curves.append(baseline)
        quality_targets.append(axes.curve(data["sizes"],values["odp"],COMPUTE))
        quality_ghosts.append(axes.curve(data["sizes"],values["baseline"],DECODE,.65))
        bf16_references.append(axes.reference(values["bf16"],"BF16"))
        # Label the endpoints and one intermediate format without crowding the curves.
        first=text(data["formats"][0],14,MUTED).next_to(
            axes.point(data["sizes"][0],values["baseline"][0]),RIGHT,buff=.12)
        last=text(data["formats"][-1],14,MUTED).next_to(
            axes.point(data["sizes"][-1],values["baseline"][-1]),DOWN,buff=.17).shift(LEFT*.35)
        mid_index=data["formats"].index("IQ2_XXS")
        middle=text("IQ2_XXS",14,MUTED).next_to(
            axes.point(data["sizes"][mid_index],values["baseline"][mid_index]),
            DOWN+RIGHT,buff=.12)
        format_labels.extend((first,middle,last))
    baseline_latency=latency_axes.curve(data["lengths"],data["latency"]["baseline"],DECODE)
    target_latency=latency_axes.curve(data["lengths"],data["latency"]["odp"],COMPUTE)
    # This is the same loading-only reference used in the plot, not a TTFT fit.
    floor=latency_axes.reference(data["floor"],"SSD loading floor",COMPUTE,below=True)
    floor[1].align_to(floor[0],RIGHT).shift(LEFT*.06)
    # BF16 leads by 0.2 s; its quick reveal overlaps the baseline curves.
    reference_reveal = AnimationGroup(*[
        AnimationGroup(Create(ref[0],lag_ratio=1),FadeIn(ref[1]),run_time=.45)
        for ref in bf16_references
    ])
    baseline_reveal = AnimationGroup(*[
        AnimationGroup(Create(curve[0]),
                       LaggedStart(*[FadeIn(dot) for dot in curve[1]],lag_ratio=.15))
        for curve in [*quality_curves,baseline_latency]
    ],Create(floor[0],lag_ratio=1),FadeIn(floor[1]),
        *[FadeIn(label) for label in format_labels],run_time=.75)
    scene.play(reference_reveal,Succession(Wait(.2),baseline_reveal))
    scene.wait(1.2)

    # Same x coordinates: adding a prefiller does not move the weight footprint.
    latency_ghost=latency_axes.curve(data["lengths"],data["latency"]["baseline"],DECODE,.65)
    gain_arrows=[]
    for axes,bench in zip(quality_axes,("mmlu_pro","mmmu")):
        values=data["quality"][bench]
        a=axes.point(data["sizes"][0],values["baseline"][0])
        b=axes.point(data["sizes"][0],values["odp"][0])
        gain_arrows.append(Arrow(a,b,buff=.12,color=COMPUTE,stroke_width=2,tip_length=.13))
    scene.play(*[Transform(c,t) for c,t in zip(quality_curves,quality_targets)],
               Transform(baseline_latency,target_latency),
               *[FadeIn(ghost) for ghost in [*quality_ghosts,latency_ghost]],
               *[FadeIn(a) for a in gain_arrows],run_time=3.8,rate_func=smooth)
    callouts=[]
    for axes,bench in zip(quality_axes,("mmlu_pro","mmmu")):
        values=data["quality"][bench]
        gain=values["odp"][0]-values["baseline"][0]
        callouts.append(text(f"IQ1_S: +{gain:.1f} points",21,COMPUTE).move_to([axes.left+1.9,-3.12,0]))
    i=data["lengths"].index(8192)
    old,new=data["latency"]["baseline"][i],data["latency"]["odp"][i]
    callouts.append(text(f"8K: {old:.2f}s → {new:.2f}s",20,COMPUTE).move_to([5.10,-3.12,0]))
    scene.play(*[FadeIn(c,shift=UP*.08) for c in callouts],run_time=.35)
    scene.wait(3)


class ODPResults(MovingCameraScene):
    def construct(self):
        animate_results(self)


class ODPWithResults(ODP):
    # Combined social video: X/Twitter's black Lights out background.
    # The separate blog scenes retain Hugging Face's dark background.
    background_color = "#000000"

    def construct(self):
        super().construct()
        self.play(*[FadeOut(m) for m in list(self.mobjects) if m is not self.branding],run_time=.6)
        animate_results(self)
