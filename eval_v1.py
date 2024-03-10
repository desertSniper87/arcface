import os

import torch
from resnet import ResNet18, ResNet50
import torchvision.transforms as transforms
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("-i", "--id", type=int, required=True, help="Name of Image id")
parser.add_argument("-w", "--weight", type=str, required=False, choices=['best', 'checkpoint'], default='best', help="Choice of checkpoints")
parser.add_argument("-p", "--perclass", type=bool, required=False, default=False, help="View per class accuracy")
parser.add_argument("-t", "--total", type=bool, required=False, default=False, help="View Total Accuracy")
parser.add_argument("-r", "--result", type=bool, required=False, default=False, help="View Total Accuracy")
args = parser.parse_args()

# Load the trained model
model = ResNet50(num_features=5013)
if args.weight == 'best':
    model.load_state_dict(torch.load('rn50_v1/best_weights.pth'))
else:
    model.load_state_dict(torch.load('rn50_v1/checkpoint.pth'))
model.eval()

if torch.cuda.is_available():
    device_id = "cuda"
elif torch.backends.mps.is_available():
    device_id = "mps"
else:
    device_id = "cpu"

# Set the device
device = torch.device(device_id)
model = model.to(device)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5677, 0.5188, 0.4885], std=[0.2040, 0.2908, 0.2848])
])

# Load the image
dir = f'/root/face-rnd/dat/personai_icartoonface_rectrain/icartoonface_rectrain/personai_icartoonface_rectrain_{args.id:05d}'

positive, total = 0, 0
result = {}

for filename in os.listdir(dir):
    image = Image.open(f'{dir}/{filename}').convert('RGB')

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class index
    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    if predicted_class not in result:
        result[predicted_class] = 1
    else:
        result[predicted_class] += 1


    if args.perclass:
        print(f'Predicted class: {predicted_class}')

    if predicted_class == args.id:
        positive += 1

    total += 1

print(result)

if args.total:
    print(f"{args.id}\t{total}\t{positive / total * 100}")
