import os
from os.path import join # Explicitly import join
import torch.utils.data as data
from PIL import Image # Import PIL.Image

# Define common image file extensions
FileNameEnd = ('.jpg', '.jpeg', '.png', '.JPEG', '.PNG', '.gif', '.bmp', '.tiff')

class ImageFolder(data.Dataset):
    def __init__(self, root, subdir='train', transform=None):
        super(ImageFolder,self).__init__()

        self.transform = transform  #定义图片转换的类型
        self.image = []     #定义存储图片的列表

        # 首先需要得到训练图片的最终路径，用来读取图片，同时需要得到图片对应的文件夹的名称，最为标签数据
        # 因此在制作数据集之前，图片存放路径及各个文件夹的命名需要规范
        train_dir_for_classes = join(root, 'train')  # Use 'train' subdir to define classes and indices consistently

        # 获取训练文件夹的路径后，train文件夹下面为各种标签命名的文件夹，读取名称作为标签数据
        # sorted可以用来根据名称对读取后的数据排序，得到列表数据
        if not os.path.isdir(train_dir_for_classes):
            raise FileNotFoundError(f"Training directory for class discovery not found: {train_dir_for_classes}")
        self.class_names = sorted(os.listdir(train_dir_for_classes))
        if not self.class_names:
            raise ValueError(f"No class folders found in {train_dir_for_classes}")

        # 然后将class_names排序，变成字典，并将序号值与文件夹名称调换位置，使得文件夹名称变为字典的keys数据，数字类型的序号变为values数据
        self.names2index = {v: k for k, v in enumerate(self.class_names)}

        # 以上算是制作标签数据的完成，之后需要根据训练、验证、测试数据来具体分析
        # 大致的思路是，获取图片具体路径，并将其与标签一一对应，得到多个数组，存入self.image中制作成列表
        # 比如self.image[1]可以检索到第二张图片的路径，以及第二张图片的标签形成的数组
        
        # Load images from the specified subdir (train, val, test, etc.)
        current_subdir_path = join(root, subdir)
        if not os.path.isdir(current_subdir_path):
            print(f"Warning: Subdirectory {current_subdir_path} not found. Dataset will be empty for this subdir.")
            return # self.image will remain empty

        for label_name in self.class_names: # Iterate through class names derived from 'train' dir
            class_dir = join(current_subdir_path, label_name)
            if not os.path.isdir(class_dir):
                # This class might not exist in the current subdir (e.g., a class in train might not be in val)
                # print(f"Warning: Class directory {class_dir} not found in {current_subdir_path}. Skipping.")
                continue

            # os.walk的用法，遍历文件夹，获取文件的路径，子文件夹的名称，以及文件的名称
            # 其中directory为文件夹的初始路径，_表示子文件夹名称，names则是文件名称
            # 需要根据具体情况进行修改
            for directory, _, file_names in os.walk(class_dir):
                for name in file_names:
                    filename = join(directory, name)
                    if filename.endswith(FileNameEnd):
                        # 注意此处的双括号，append()可以把数据加到列表后，此处需要的是把数组加进去，因此有append(())
                        self.image.append((filename, self.names2index[label_name]))

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image)

    def __getitem__(self, index):
        """Return a single image and its label from the dataset."""
        image_path, label = self.image[index]
        
        try:
            img = Image.open(image_path).convert('RGB') # Load image and convert to RGB
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Optionally, return a placeholder or raise the exception
            # For now, let's make it return None, None and let DataLoader handle it or skip
            if self.transform:
                # Try to return something of the expected type if a transform exists
                # This is a bit of a hack; robust error handling is complex.
                try:
                    return self.transform(Image.new('RGB', (224, 224), (255,0,0))), torch.tensor(-1) # Dummy red image and invalid label
                except:
                    pass # if transform itself errors on dummy data
            return None, None 

        if self.transform:
            img = self.transform(img)
        
        return img, label