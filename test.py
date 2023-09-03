import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gc
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from Classes.picturedrawer import PictureDrawer
from Classes.picturedrawer import My_Dataset
from Classes.picturedrawer import ShapeDetector
from Classes.picturedrawer import UNet


def train_cnn(epoch, loader, batch_size=32):
    # Функция обечения нейросети
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_score': []}
    for epoch in range(epoch):
        running_loss = 0.0
        model.train(True)
        dict_names = {}
        for i, batch in enumerate(loader):

            X_batch, y_batch = batch['image'], batch['label']
            image_name, file_name = batch['image_name'], batch['file_name']
            names = batch['names']
            for name in names:
                for fig_name in name:
                    if fig_name not in dict_names.keys():
                        dict_names[fig_name] = 1
                    else:
                        dict_names[fig_name] += 1
            optimizer.zero_grad()
            outputs = model(X_batch.to(device))
            # outputs_model = model(X_batch.to(device))
            # outputs = outputs_model.view(-1, 5, 4)
            loss = criterion(outputs, y_batch.to(device).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #         exp_lr_scheduler.step()
        history['epoch'].append(epoch)
        history['train_loss'].append(running_loss)
        # history['val_loss'].append(val_loss.item())
        # history['val_score'].append(val_score.item())
        clear_output(wait=True)
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
        count = 0
        for key, value in dict_names.items():
            print('Количество', key, 'в эпохе обучения:', value)
            count += value
        print('Количество фигур в эпохе обучения:', count)
        plt.figure(figsize=(18, 6))
        if len(X_batch) > 6:
            end = 6
        else:
            end = len(X_batch)
        for i in range(end):
            plt.subplot(2, 6, i + 1)
            plt.axis("off")
            plt.imshow(np.rollaxis(X_batch[i].numpy(), 0, 3))
            plt.title(image_name[i])
            plt.subplot(2, 6, i + 7)
            plt.axis("off")
            image = Image.new("RGB", (256, 256))
            draw = ImageDraw.Draw(image)
            shapes = np.array(y_batch[i])
            for j in range(5):
                x1, y1 = shapes[j][0], shapes[j][1]
                x2, y2 = x1 + shapes[j][2], y1 + shapes[j][3]
                draw.polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], outline='white')
            #         shapes = outputs[i].detach().numpy()
            shapes = outputs[i].cpu()
            for j in range(5):
                x1, y1 = shapes[j][0], shapes[j][1]
                x2, y2 = x1 + shapes[j][2], y1 + shapes[j][3]
                draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline='red')
            plt.imshow(image)
            plt.title(file_name[i])
        plt.show()
    return history


def eval_cnn(loader, batch_size=32):
    # Функция проверки работы нейронной сети
    iou_min = 1000
    iou_max = 0
    iou = []
    iou_fp = 0
    precision = []
    recall = []
    model.eval()
    dict_names = {}
    count = 0
    for i, batch in enumerate(loader):

        X_batch, y_batch = batch['image'], batch['label']
        image_name, file_name = batch['image_name'], batch['file_name']
        names = batch['names']
        for name in names:
            for fig_name in name:
                if fig_name not in dict_names.keys():
                    dict_names[fig_name] = 1
                else:
                    dict_names[fig_name] += 1
        outputs = model(X_batch.to(device))

        if len(X_batch) > 6:
            end = 6
        else:
            end = len(X_batch)
        for j in range(end):
            shapes_real = y_batch[j].cpu().detach().numpy()
            shapes_predicted = outputs[j].cpu().detach().numpy()
            for k in range(5):
                x1, y1 = shapes_real[k][0], shapes_real[k][1]
                x2, y2 = x1 + shapes_real[k][2], y1 + shapes_real[k][3]
                polygon1 = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
                polygon1_list = [x1, y1, x1, y2, x2, y2, x2, y1]
                x1, y1 = shapes_predicted[k][0], shapes_predicted[k][1]
                x2, y2 = x1 + shapes_predicted[k][2], y1 + shapes_predicted[k][3]
                polygon2 = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
                polygon2_list = [x1, y1, x1, y2, x2, y2, x2, y1]
                intersect = polygon1.intersection(polygon2).area
                union = polygon1.union(polygon2).area
                iou_moment = intersect / union
                if iou_moment > 0 and iou_moment < 0.5:
                    iou_fp += 1
                if iou_moment > 0.5:
                    iou.append(iou_moment)
                if iou_moment < iou_min:
                    iou_min = iou_moment
                    min_image = X_batch[j]
                    min_poligon_real = polygon1_list
                    min_poligon_predicted = polygon2_list
                if iou_moment > iou_max:
                    iou_max = iou_moment
                    max_image = X_batch[j]
                    max_poligon_real = polygon1_list
                    max_poligon_predicted = polygon2_list
                count += 1
    tp = len(iou)
    fp = iou_fp - tp

    # Calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / count
    return (precision,
            recall,
            min_poligon_real,
            min_poligon_predicted,
            max_poligon_real,
            max_poligon_predicted,
            min_image,
            max_image,
            iou_min,
            iou_max,
            iou
            )


# Первый этап: генерация 100 рандомных картинок в папку first_stage
drawer = PictureDrawer(5, hexagon_status=True, random_status=True)
drawer.generate_images(100, 'first_stage', update=False)

# Второй этап: обучение модели на сгенерированных на лету фигурах
# train_data = My_Dataset(transform_train, fly=False)
train_data = My_Dataset(fly=True)
drawer.generate_images(1000, 'first_test', update=False)
drawer.generate_images(1000, 'second_test', update=False)

train_data = My_Dataset(fly=True)

test_1_data = My_Dataset(folder='first_test', fly=True)
test_2_data = My_Dataset(folder='second_test', fly=False)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_1_loader = torch.utils.data.DataLoader(test_1_data, batch_size=batch_size, shuffle=False)
test_2_loader = torch.utils.data.DataLoader(test_2_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.cuda.empty_cache()
gc.collect()

model = ShapeDetector().to(device)
# model = My_CNN().to(device)
# model = FigureDetector().to(device)
# model = UNet().to(device)

criterion = nn.MSELoss()
# criterion = Customloss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

history = train_cnn(20, train_loader, batch_size=batch_size)
