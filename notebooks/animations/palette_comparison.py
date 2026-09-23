"""Static palette choices; does not change either animation's colors.

Render with Manim -s -r 1920,1200 ... PaletteComparison.
Brand colors come from 01_BRAND_ASSETS/charts/chart-style-reference.md
and the NVIDIA26_v2 color scheme inside the supplied .potx files.
"""

import numpy as np
from manim import (
    Arrow, DashedLine, Dot, LEFT, Line, Rectangle, RoundedRectangle,
    Scene, VGroup, VMobject, config,
)

from odp import text
from pareto_data import load_results

config.frame_width = 32
config.frame_height = 20
config.background_color = "#202020"

SCHEMES = [
    dict(title="A · Current", note="Existing colors · comparison reference",
         bg="#101720", panel="#172230", ink="#E9EFF5", muted="#91A0B3",
         edge="#37465A", baseline="#4D9BE8", compute="#E45C68",
         load="#EFAB46", cache="#65C99A"),
    dict(title="B · NVIDIA Dark / Focus", note="Recommended · gray baseline, green improvement",
         bg="#000000", panel="#313131", ink="#FFFFFF", muted="#F7F7F7",
         edge="#757575", baseline="#757575", compute="#76B900",
         load="#EF9100", cache="#1DBBA4"),
    dict(title="C · NVIDIA Dark / Color", note="More color · blue baseline, green improvement",
         bg="#000000", panel="#313131", ink="#FFFFFF", muted="#F7F7F7",
         edge="#757575", baseline="#0074DF", compute="#76B900",
         load="#EF9100", cache="#1DBBA4"),
    dict(title="D · NVIDIA Light / Focus", note="Light alternative · gray baseline, green improvement",
         bg="#FFFFFF", panel="#F7F7F7", ink="#000000", muted="#313131",
         edge="#757575", baseline="#757575", compute="#76B900",
         load="#EF9100", cache="#1DBBA4"),
]


def card(scheme, data):
    s = scheme
    group = VGroup(Rectangle(width=15.5,height=9.5,stroke_width=1,
                            stroke_color="#757575",fill_color=s["bg"],fill_opacity=1))

    def label(value,x,y,size=19,color=None,weight="NORMAL",left=False):
        item=text(value,size,color or s["ink"],weight=weight)
        item.move_to([x,y,0])
        if left:
            item.align_to(np.array([x,y,0]),LEFT)
        group.add(item)
        return item

    label(s["title"],-7.2,4.10,30,weight="MEDIUM",left=True)
    label(s["note"],-7.2,3.52,18,s["muted"],left=True)
    label("Offloaded Disaggregated Prefill",-3.5,2.65,23,weight="MEDIUM")
    label("MMLU-Pro",3.9,2.65,23,weight="MEDIUM")
    label("Quality at the same device weight size",3.9,2.17,16,s["muted"])

    # Small snapshot of the existing mechanism: load / compute overlap and KV.
    for x,name in [(-5.7,"Prefill"),(-3.45,"KV"),(-1.25,"Decode")]:
        label(name,x,1.92,18,weight="MEDIUM")
    for row,y in enumerate((.95,-.05,-1.05)):
        pre=RoundedRectangle(width=1.9,height=.60,corner_radius=.07,
            stroke_width=1.2,stroke_color=s["edge"],fill_color=s["panel"],fill_opacity=1)
        pre.move_to([-5.7,y,0]);group.add(pre)
        if row<2:
            color=s["load"] if row==0 else s["compute"]
            bar=Rectangle(width=1.78*(.45 if row==0 else .75),height=.48,
                          stroke_width=0,fill_color=color,fill_opacity=.50)
            bar.move_to(pre).align_to(pre,LEFT).shift(np.array([.06,0,0]))
            group.add(bar)
        label(("Loading","Computing","Offloaded")[row],-5.7,y,17,
              s["muted"] if row==2 else s["ink"])
        kv=RoundedRectangle(width=1.2,height=.60,corner_radius=.07,
            stroke_width=1.2,stroke_color=s["cache"] if row==2 else s["edge"],
            fill_color=s["cache"] if row==2 else s["panel"],
            fill_opacity=.25 if row==2 else 1).move_to([-3.45,y,0])
        group.add(kv);label("KV",-3.45,y,16)
        dec=RoundedRectangle(width=1.9,height=.60,corner_radius=.07,
            stroke_width=1.4,stroke_color=s["baseline"],fill_color=s["baseline"],
            fill_opacity=.20).move_to([-1.25,y,0])
        group.add(dec);label("Resident",-1.25,y,17)
        for a,b in [(pre.get_right(),kv.get_left()),(kv.get_right(),dec.get_left())]:
            group.add(Arrow(a,b,buff=.07,color=s["edge"],stroke_width=1.5,
                            max_tip_length_to_length_ratio=.20))
    for x in (-5.7,-1.25):
        for y in (-1.05,-.05):
            group.add(Arrow([x,y+.33,0],[x,y+.65,0],buff=.015,
                            color=s["cache"],stroke_width=1.5))
    label("NVFP4, offloaded",-5.7,-1.65,15,s["muted"])
    label("Always on device",-1.25,-1.65,15,s["muted"])

    # Actual MMLU-Pro Pareto values; common axes in every option.
    sizes=data["sizes"]
    values=data["quality"]["mmlu_pro"]
    def point(x,y):
        return np.array([1.1+(np.log(x)-np.log(6))/(np.log(11.5)-np.log(6))*5.6,
                         -1.45+(y-20)/72*3.1,0])
    for y in (20,40,60,80):
        a,b=point(6,y),point(11.5,y)
        group.add(Line(a,b,color=s["edge"],stroke_width=.65,stroke_opacity=.5))
        label(str(y),.73,a[1],15,s["muted"])
    for x in (6,8,10):
        p=point(x,20)
        label(str(x),p[0],p[1]-.26,15,s["muted"])
    group.add(DashedLine(point(6,values["bf16"]),point(11.5,values["bf16"]),
                        color=s["edge"],dash_length=.06,stroke_width=1))
    label("BF16",1.5,point(6,values["bf16"])[1]+.19,15,s["muted"])
    for arm,color in [("baseline",s["baseline"]),("odp",s["compute"])]:
        points=[point(x,y) for x,y in zip(sizes,values[arm])]
        group.add(VMobject().set_points_as_corners(points).set_stroke(color,width=3))
        group.add(*[Dot(p,radius=.045,color=color) for p in points])
    label("Device weights, GB",3.9,-2.2,17,s["muted"])
    for x,color,name in [(1.65,s["baseline"],"Weight-only"),(4.4,s["compute"],"+ NVFP4")]:
        group.add(Line([x-.5,-2.75,0],[x-.15,-2.75,0],color=color,stroke_width=3))
        label(name,x,-2.75,17,left=True)

    for i,(key,name) in enumerate([("baseline","Decode"),("compute","Compute"),
                                  ("load","Load"),("cache","KV / activations")]):
        x=-6.95+i*3.65
        group.add(Rectangle(width=.33,height=.33,stroke_width=0,
                  fill_color=s[key],fill_opacity=1).move_to([x,-3.72,0]))
        label(name,x+.31,-3.58,16,left=True)
        label(s[key],x+.31,-3.94,14,s["muted"],left=True)
    return group


class PaletteComparison(Scene):
    def construct(self):
        data=load_results()
        for s,center in zip(SCHEMES,[(-8,5),(8,5),(-8,-5),(8,-5)]):
            self.add(card(s,data).move_to([*center,0]))
