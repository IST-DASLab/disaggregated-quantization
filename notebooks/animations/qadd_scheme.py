"""Static QADD schematic for the blog, matching the Manim animations.

Render: manim -s -r 1920,960 ... qadd_scheme.py QADDScheme
Prompt/response routing occurs within each student linear; the teacher is frozen.
Solid arrows show forward flow, dashed arrows show gradients.
"""

import numpy as np
from manim import Arrow, DashedLine, LEFT, Line, RIGHT, RoundedRectangle, Scene, VGroup, config

from odp import COMPUTE, DECODE, EDGE, INK, MUTED, PANEL, text

config.frame_width=16
config.frame_height=8


def box(center,width,height,color=EDGE,fill=PANEL,opacity=.40):
    return RoundedRectangle(width=width,height=height,corner_radius=.10,
        stroke_color=color,stroke_width=1.3,fill_color=fill,fill_opacity=opacity).move_to([*center,0])


def path(points,color=INK,dashed=False,head=True):
    points=[np.array([*point,0]) for point in points]
    group=VGroup()
    for i,(a,b) in enumerate(zip(points,points[1:])):
        last=i==len(points)-2
        if dashed:
            # A solid arrow tip keeps gradient direction readable at small size.
            end=b-(b-a)/np.linalg.norm(b-a)*.12 if last and head else b
            group.add(DashedLine(a,end,color=color,stroke_width=1.4,dash_length=.06))
            if last and head:
                group.add(Arrow(end-(b-a)/np.linalg.norm(b-a)*.06,b,buff=0,
                                color=color,stroke_width=1.4,tip_length=.10))
        elif last and head:
            group.add(Arrow(a,b,buff=0,color=color,stroke_width=1.6,tip_length=.11))
        else:
            group.add(Line(a,b,color=color,stroke_width=1.6))
    return group


def cells(colors):
    return VGroup(*[box((0,0),.13,.32,c,c,.32) for c in colors]).arrange(RIGHT,buff=.035)


class QADDScheme(Scene):
    def construct(self):
        self.add(text("Quantization-Aware Distillation with Disaggregation",30,
                      weight="MEDIUM").move_to([0,3.5,0]))

        # One SFT sequence goes to both student and frozen teacher.
        prompt_words=["What","is","the","capital","of","France","?"]
        response_words=["It's","Paris","."]
        chips=[]
        for word,color in [(w,COMPUTE) for w in prompt_words]+[(w,DECODE) for w in response_words]:
            label=text(word,19)
            tile=box((0,0),label.width+.17,.47,color,color,.22)
            chips.append(VGroup(tile,label.move_to(tile)))
        sequence=VGroup(*chips).arrange(RIGHT,buff=.04).move_to([-2.2,2.58,0])
        batch=box((-2.2,2.36),10.5,1.15)
        self.add(batch,sequence)
        for subset,name,color in [(chips[:7],"Prompt",COMPUTE),(chips[7:],"Response",INK)]:
            group=VGroup(*subset)
            self.add(text(name,16,color).move_to([group.get_x(),2.06,0]))

        student=box((-2.55,-1.05),9.8,4.75)
        self.add(student,text("Student",23,weight="MEDIUM").move_to([-5.95,.96,0]),
                 text("Phase routing in each quantized linear",17,MUTED).move_to([-.75,.96,0]))

        teacher=box((5.50,.65),3.6,1.30)
        loss=box((5.50,-1.85),3.6,1.5)
        self.add(teacher,loss,
                 text("Teacher",24,weight="MEDIUM").move_to([5.50,.90,0]),
                 text("Frozen BF16",18,MUTED).move_to([5.50,.40,0]),
                 text("Distillation Loss",23,weight="MEDIUM").move_to([5.50,-1.43,0]),
                 text("KL(teacher ∥ student)",19).move_to([5.50,-1.90,0]),
                 text("Response targets only",17,MUTED).move_to([5.50,-2.35,0]))
        self.add(path([(batch.get_right()[0],2.58),(5.5,2.58),(5.5,1.30)]),
                 path([(5.5,0),(5.5,-1.10)]))

        lanes=[(-.35,COMPUTE,"Prefill",7),(-2.05,DECODE,"Decode",3)]
        split_x,merge_x=-5.60,.40
        mid_y=-1.20
        self.add(path([(-7.10,batch.get_bottom()[1]),(-7.10,mid_y+.07),(split_x,mid_y+.07)],head=False),
                 path([(split_x,mid_y-.07),(-6.95,mid_y-.07)],dashed=True))
        self.add(text("Forward",14,MUTED).move_to([-6.45,mid_y+.32,0]),
                 text("Backward",14,MUTED).move_to([-6.45,mid_y-.35,0]))
        for y,color,name,count in lanes:
            branch=box((-1.80,y),2.8,.90,color,color,.17)
            lane_tokens=cells([color]*count).move_to([-4.40,y,0])
            self.add(branch,lane_tokens,
                     text(name,22,weight="MEDIUM").move_to([-1.80,y+.16,0]),
                     text("pathway",17,MUTED).move_to([-1.80,y-.21,0]))
            left,right=lane_tokens.get_left()[0],lane_tokens.get_right()[0]
            self.add(path([(split_x,mid_y+.07),(split_x,y+.07),(left-.06,y+.07)],color),
                     path([(right+.06,y+.07),(-3.20,y+.07)],color),
                     path([(-3.20,y-.07),(right+.06,y-.07)],color,dashed=True),
                     path([(left-.06,y-.07),(split_x+.12,y-.07),(split_x+.12,mid_y-.07)],color,dashed=True,head=False),
                     path([(-.40,y+.07),(merge_x,y+.07),(merge_x,mid_y+.07)],color,head=False),
                     path([(merge_x+.12,mid_y-.07),(merge_x+.12,y-.07),(-.40,y-.07)],color,dashed=True))

        merged=cells([COMPUTE]*7+[DECODE]*3).move_to([1.47,mid_y,0])
        self.add(merged,
                 path([(merge_x,mid_y+.07),(merged.get_left()[0]-.03,mid_y+.07)]),
                 path([(merged.get_left()[0]-.03,mid_y-.07),(merge_x+.12,mid_y-.07)],dashed=True),
                 path([(merged.get_right()[0]+.05,mid_y+.07),(2.90,mid_y+.07),(2.90,-1.72),(3.70,-1.72)]),
                 path([(3.70,-2.03),(2.70,-2.03),(2.70,mid_y-.07),(merged.get_right()[0]+.05,mid_y-.07)],dashed=True))
        self.add(text("…",24,MUTED).move_to([2.98,-.71,0]),
                 text("Same token order",15,MUTED).move_to([1.43,-1.70,0]),
                 text("One Forward–Backward Pass",21,weight="MEDIUM").move_to([-2.55,-2.83,0]),
                 text("Response gradients also reach prefill through the prompt's cached representations.",18,MUTED)
                 .move_to([0,-3.73,0]))
