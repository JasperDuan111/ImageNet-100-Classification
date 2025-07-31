import json
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

with open('data/Labels.json', 'r') as f:
    labels = json.load(f)

id_to_class = {str(k): v for k, v in labels.items()}

base_train_dir = 'data'
val_dir = os.path.join(base_train_dir, 'val.X')
train_dirs = [os.path.join(base_train_dir, f'train.X{i}') for i in range(1, 5)]

class ImageNetDataset(Dataset):
    def __init__(self, data_dirs, labels_dict, transform=None):
        self.transform = transform
        self.labels_dict = labels_dict
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(labels_dict.keys())}
        self.image_paths = []
        self.labels = []
        self._scan_images(data_dirs)
    
    def _scan_images(self, data_dirs):
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: Directory {data_dir} does not exist")
                continue
                
            for class_folder in os.listdir(data_dir):
                class_path = os.path.join(data_dir, class_folder)
                if os.path.isdir(class_path) and class_folder in self.class_to_idx:
                    for image_file in os.listdir(class_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_path = os.path.join(class_path, image_file)
                            self.image_paths.append(image_path)
                            self.labels.append(self.class_to_idx[class_folder])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
        
        return image, label

def get_data():

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageNetDataset(
        data_dirs=train_dirs,
        labels_dict=labels,
        transform=train_transform
    )
    
    test_dataset = ImageNetDataset(
        data_dirs=[val_dir],
        labels_dict=labels,
        transform=test_transform
    )
    
    # train_data_size = len(train_dataset)
    # test_data_size = len(test_dataset)
    
    # print(f"Train dataset size: {train_data_size}")
    # print(f"Test dataset size: {test_data_size}")
    # print(f"Number of classes: {len(labels)}")
    

    # 如果性能足够，可以将 batch_size 调整为更大的值
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=4,  
        pin_memory=True  
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,  
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_data_loader, test_data_loader