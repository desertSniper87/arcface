import os

import torch
from resnet import ResNet18, ResNet50
import torchvision.transforms as transforms
from PIL import Image

# Load the trained model
model = ResNet50(num_features=5013)
model.load_state_dict(torch.load('rn50_v1/best_weights.pth'))
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

imgpaths = []
imgpath_classids = []
correct_num, total_num = 0, 0

dir = '/root/face-rnd/dat/personai_icartoonface_rectest/icartoonface_rectest'

def get_predicted_class(img_path):
    image = Image.open(f'{dir}/{img_path}').convert('RGB')

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class index
    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()
    return predicted_class


with open(f"/root/face-rnd/dat/icartoonface_rectest_info.txt", 'r', encoding='utf-8') as f , open("report_50_v1", "w+") as f_write:
    for line in f.readlines():
        line_info = line.strip().split()
        if len(line_info) == 6:
            imgpaths.append(line_info[0])
            imgpath_classids.append(line_info[-1])
        if len(line_info) == 2:
            imgpath1, imgpath2 = line_info[0], line_info[1]
            idx1, idx2 = imgpaths.index(imgpath1), imgpaths.index(imgpath2)
            total_num += 1
            if get_predicted_class(imgpath1) == get_predicted_class(imgpath2) \
                    and imgpath_classids[idx1] == imgpath_classids[idx2]:
                print(f'ok match', end=' ')
                correct_num += 1
            elif imgpath_classids[idx1] == -1 or imgpath_classids[idx2] == -1:
                print(f'ok no match', end=' ')
                correct_num += 1
            elif get_predicted_class(imgpath1) != get_predicted_class(imgpath2) \
                    and imgpath_classids[idx1] != imgpath_classids[idx2]:
                print(f'ok no match', end=' ')
                correct_num += 1
            else:
                print(f'not ok', end=' ')
            print(f'\t{idx1}\t{idx2}\t{imgpath1}\t{imgpath_classids[idx1]}\t{get_predicted_class(imgpath1)}\t{imgpath2}\t{imgpath_classids[idx2]}\t{get_predicted_class(imgpath2)}', 100.0 * correct_num / total_num)
            f_write.write(f',{idx1},{idx2},{imgpath1},{imgpath_classids[idx1]},{get_predicted_class(imgpath1)},{imgpath2},{imgpath_classids[idx2]},{get_predicted_class(imgpath2)},' + str(100.0 * correct_num / total_num))

