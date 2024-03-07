import torch
from resnet import ResNet18
import torchvision.transforms as transforms
from PIL import Image

# Load the trained model
model = ResNet18()
model.load_state_dict(torch.load('rn18_v1/best_weights.pth'))
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
image = Image.open('/Users/bccca/dev/dat/personai_icartoonface_rectrain/icartoonface_rectrain/personai_icartoonface_rectrain_00004/personai_icartoonface_rectrain_00004_0000000.jpg')

# Preprocess the image
image_tensor = transform(image).unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    output = model(image_tensor)

# Get the predicted class index
_, predicted_class = torch.max(output, 1)
predicted_class = predicted_class.item()
print(f'Predicted class: {predicted_class}')