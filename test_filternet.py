from filternet import FilternetPredictor
from PIL import Image

checkpoint = "/home/pedro/code/photo-editing/filternet/models/p-tebcwh-batch-norm-moderate-resnet50-noconv-unfrozen.pth"

predictor = FilternetPredictor(checkpoint)
print(predictor)

from diffusers.utils import load_image
image = load_image("images/cat-crappy.png")

prediction = predictor.predict(image)

prediction.image.save("filternet-predicted.png")
print(f"Predicted filters: {prediction.filters}")

# TODO: I'm getting params close to the notebook version, but not identical
# notebook: [-0.01318, -0.07661, -0.2604, 0.15502, -0.75897, -0.39759]
# script version: {'temperature': -0.012949585914611816, 'ev': -0.06620264053344727, 'brightness': -0.2182731032371521, 'contrast': 0.1490241289138794, 'shadows': -0.7749564051628113, 'highlights': -0.3049997091293335}
# I'm suspecting the bilinear interpolation to downscale to 299x299, I think fastai does it differently than PIL