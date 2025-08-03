import os
import json
import random
from PIL import Image
import torch
from torchvision import transforms  
from model import InceptionNet

def load_imagenet_labels():
    with open('data/Labels.json', 'r') as f:
        labels = json.load(f)
    idx_to_class = {}
    for idx, (class_id, class_name) in enumerate(labels.items()):
        idx_to_class[idx] = class_name
    return idx_to_class

IMAGENET_CLASSES = load_imagenet_labels()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_random_test_image():

    val_dir = 'data/val.X'
    class_folders = [f for f in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, f))]
    random_class = random.choice(class_folders)
    class_path = os.path.join(val_dir, random_class)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random_image = random.choice(images)
    img_path = os.path.join(class_path, random_image)
    
    return img_path, random_class

img_path, true_class_id = get_random_test_image()
img = Image.open(img_path)
img.show()

with open('data/Labels.json', 'r') as f:
    original_labels = json.load(f)
true_class_name = original_labels.get(true_class_id, "Unknown")
true_class_idx = list(original_labels.keys()).index(true_class_id) if true_class_id in original_labels else None

print(f"True class ID: {true_class_id}")
print(f"True class name: {true_class_name}")
print(f"True class index: {true_class_idx}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)  # 从 [3, 224, 224] 变成 [1, 3, 224, 224]
img_tensor = img_tensor.to(device)

model_path = 'model/mynet_best_model.pth' 
model = InceptionNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()


# TOP-1 预测
print(f"\n{'='*50}")
print(f"TOP-1 PREDICTION")
print(f"{'='*50}")
print(f"Predicted class name: {IMAGENET_CLASSES[predicted_class]}")
print(f"Confidence: {probabilities[0][predicted_class].item()*100:.2f}%")
print(f"True class name: {true_class_name}")

if true_class_idx is not None:
    is_correct = predicted_class == true_class_idx
    print(f"\nPrediction Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

# TOP-5 预测
print(f"\n{'='*50}")
print(f"TOP-5 PREDICTIONS")
print(f"{'='*50}")
top_probs, top_indices = torch.topk(probabilities[0], 5)

for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    idx_int = idx.item()  
    is_true_class = " ← CORRECT ANSWER" if idx_int == true_class_idx else ""
    rank = f"#{i+1}"
    print(f"{rank:>3} {IMAGENET_CLASSES[idx_int]:<40} {prob.item()*100:>6.2f}%{is_true_class}")

if true_class_idx is not None:
    top5_correct = true_class_idx in [idx.item() for idx in top_indices]
    print(f"\nTop-5 Prediction: {'✓ CORRECT' if top5_correct else '✗ INCORRECT'}")
    if top5_correct and not is_correct:
        true_rank = [idx.item() for idx in top_indices].index(true_class_idx) + 1
        print(f"True class ranked #{true_rank} in top-5 predictions")

