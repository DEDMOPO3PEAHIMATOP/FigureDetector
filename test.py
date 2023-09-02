import torch
import torchvision.transforms as transforms

from Classes.picturedrawer import PictureDrawer
from Classes.picturedrawer import My_Dataset

# Первый этап: генерация 100 рандомных картинок в папку first_stage
drawer = PictureDrawer(5, hexagon_status=True, random_status=True)
drawer.generate_images(100, 'first_stage', update=False)

# Второй этап: обучение модели на сгенерированных на лету фигурах
# train_data = My_Dataset(transform_train, fly=False)
train_data = My_Dataset(fly=True)

batch_size = 8
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

