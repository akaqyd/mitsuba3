# %%
import os
import sys
# sys.path.append("/home/ziyizhang/Desktop/Projects/mitsuba-uni/mitsuba3/build/python")
sys.path.insert(0, "/home/ziyizhang/Desktop/Projects/mitsuba-curve/mitsuba3/build/python")

import mitsuba as mi
import drjit as dr

from typing import Union
import matplotlib.pyplot as plt
import numpy as np


print("Loaded Mitsuba from: ", os.path.dirname(mi.__file__))
print("Loaded Mitsuba from: ", os.path.dirname(dr.__file__))

mi.set_variant("llvm_ad_rgb")
# mi.set_variant("scalar_rgb")

# dr.set_log_level(dr.LogLevel.Trace)
# mi.set_log_level(mi.LogLevel.Trace)

# scene_path = "/home/qiyuan/sp/mitsuba3/notebooks/bspline_curve.xml"
# scene = mi.load_file("cornell_curve.xml")

# %%
T = mi.ScalarTransform4f
scene_dict = mi.cornell_box()
scene_dict["curve1"] = {
    "type": "bspline",
    "filename": "data_samples/curve1.txt",
    "to_world": T.translate([0, 0, -0.2]) @ T.rotate([1, 0, 0], -20) @ T.scale(0.2),
    "bsdf": {
        "type": "ref",
        "id": "white"
    }
}
scene = mi.load_dict(scene_dict)

# %%
image = mi.render(scene, spp=256)

# %%
mi.Bitmap(image)

# %%


# %%



