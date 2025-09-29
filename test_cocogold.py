from cocogold_pipeline import load_cocogold_pipeline, run_cocogold_inference
from PIL import Image

# checkpoint = "iter_018000"
checkpoint = "iter_008000"
unet_checkpoint = "/home/pedro/code/hf/diffusers/marigold-segmentation/Marigold/output/overlapped/25_06_11-13_00_18-train_cocogold/checkpoint/" + checkpoint
sd_checkpoint = "/home/pedro/code/hf/diffusers/marigold-segmentation/checkpoints/stable-diffusion-2"

pipe = load_cocogold_pipeline(unet_checkpoint, sd_checkpoint)
print(pipe)

inputs = ("images/clock-original.png", "clock")
# inputs = ("images/bike-horse.png", "horse")

from diffusers.utils import load_image
image = load_image(inputs[0])

predicted_image, mask_image = run_cocogold_inference(pipe, image, inputs[1], show_progress=True)
predicted_image.save("predicted.png")
mask_image.save("mask.png")
