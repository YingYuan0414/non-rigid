import numpy as np
import os
from PIL import Image, ImageSequence

path = "/home/jacinto/data/outputs/viz/tax3dv2_double/val/0_success_diffusion.gif"
out_path = "/home/jacinto/data/outputs/viz/tax3dv2_double/viz/"

# multimodal, rigid, non-rigid, precise
frames = []
with Image.open(path) as gif:
    for frame in ImageSequence.Iterator(gif):
        frames.append(np.array(frame.convert("RGBA")))
    frames = np.array(frames)


for i in range(frames.shape[0]):
    frame = frames[i]
    frame = Image.fromarray(frame)
    frame.save(os.path.join(out_path, f"{i}.png"))

breakpoint()