from torchvision import transforms

def get_data_transforms():
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        tranforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        transforms.ToTensor()
    ])

    return data_transform