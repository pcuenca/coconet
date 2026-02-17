import argparse
from pathlib import Path

from filternet import FilternetPredictor
from PIL import Image

checkpoint = "/home/pedro/code/photo-editing/filternet/models/p-tebcwh-batch-norm-moderate-resnet50-noconv-unfrozen.pth"

predictor = FilternetPredictor(checkpoint)
print(predictor)

parser = argparse.ArgumentParser(description="Run Filternet on a single image.")
parser.add_argument("image", help="Path to input image.")
args = parser.parse_args()

input_path = Path(args.image)
image = Image.open(input_path).convert("RGB")

square_for_prediction = image.resize(
    (predictor.image_size, predictor.image_size), Image.Resampling.BILINEAR
)
prediction = predictor.predict(square_for_prediction)
intensities = [float(prediction.filters[name]) for name in predictor.filter_names]
output_image = predictor.apply_filters(image, intensities)

if input_path.suffix:
    output_path = input_path.with_name(f"{input_path.stem}-filternet{input_path.suffix}")
else:
    output_path = input_path.with_name(f"{input_path.name}-filternet")

output_image.save(str(output_path))
print(f"Predicted filters: {prediction.filters}")
print(f"Saved prediction to: {output_path}")

# TODO: I'm getting params close to the notebook version, but not identical
# notebook: [-0.01318, -0.07661, -0.2604, 0.15502, -0.75897, -0.39759]
# script version: {'temperature': -0.012949585914611816, 'ev': -0.06620264053344727, 'brightness': -0.2182731032371521, 'contrast': 0.1490241289138794, 'shadows': -0.7749564051628113, 'highlights': -0.3049997091293335}
# I'm suspecting the bilinear interpolation to downscale to 299x299, I think fastai does it differently than PIL
