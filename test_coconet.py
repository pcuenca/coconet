from cocogold_pipeline import load_cocogold_pipeline
from coconet import apply_filternet_with_cocogold, apply_filternet_with_cocogold_bbox
from filternet import FilternetPredictor

from diffusers.utils import load_image

inputs = ("images/clock-original.png", "horse")

# cocogold checkpoint
# cocogold_checkpoint = "iter_018000"
cocogold_checkpoint = "iter_008000"
unet_checkpoint = "/home/pedro/code/hf/diffusers/marigold-segmentation/Marigold/output/overlapped/25_06_11-13_00_18-train_cocogold/checkpoint/" + cocogold_checkpoint
sd_checkpoint = "/home/pedro/code/hf/diffusers/marigold-segmentation/checkpoints/stable-diffusion-2"
pipe = load_cocogold_pipeline(unet_checkpoint, sd_checkpoint)

# filternet checkpoint
filternet_model = "/home/pedro/code/photo-editing/filternet/models/p-tebcwh-batch-norm-moderate-resnet50-noconv-unfrozen.pth"
filternet = FilternetPredictor(filternet_model)

image = load_image(inputs[0])
prompt = inputs[1]

# Apply filternet to full image, then blend
result = apply_filternet_with_cocogold(
    image=image,
    prompt=prompt,
    filternet=filternet,
    cocogold_pipeline=pipe,
    mask_feather_radius=5.0,
)

result.composited.save("blended-full.png")
result.mask.save("mask-full.png")
result.cocogold_prediction.save("cocogold-predicted-full.png")
print(f"Filternet filters (full image): {result.filternet.filters}")

# Apply filternet to bounding box around mask
result = apply_filternet_with_cocogold_bbox(
    image=image,
    prompt=prompt,
    filternet=filternet,
    cocogold_pipeline=pipe,
    bbox_padding=12,
    mask_feather_radius=5.0,
)

result.composited.save("blended-bbox.png")
result.mask.save("mask-bbox.png")
result.cocogold_prediction.save("cocogold-predicted-bbox.png")
print(f"Filternet filters (bbox): {result.filternet.filters}")
print(f"Bbox: {result.bbox}")
