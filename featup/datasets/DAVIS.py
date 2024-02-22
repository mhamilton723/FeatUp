from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset


class DAVIS(Dataset):
    def __init__(self, root, video_name, transform=None):
        """
        Args:
            root (string): Directory with all the videos.
            video_name (string): Name of the specific video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root, "DAVIS/JPEGImages/480p/", video_name)
        self.frames = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.frames[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"img": image, "img_path": img_path}


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    davis_dataset = DAVIS(root='/pytorch-data', video_name="motocross-jump", transform=transform)

    frames = davis_dataset[0]

    print("here")
