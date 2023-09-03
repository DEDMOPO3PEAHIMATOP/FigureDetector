import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gc
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

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

    # Вычисляем precision и recall
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


def transfer_learning(epoch, train_epoch, status):
    '''
    Функция дообучения сети
    '''
    train_data = My_Dataset(fly=False, folder='work')
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    torch.cuda.empty_cache()
    gc.collect()
    model = ShapeDetector().to(device)
    model.load_state_dict(torch.load('model_simple_start_learning.pth', map_location=torch.device(device)))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = train_cnn(train_epoch, train_loader, batch_size=32)
    eval_output = eval_cnn(test_1_loader, batch_size=32)
    status
    status = status.append(
        {
            'test': f'{epoch}_test', \
            'precision': eval_output[0], 'recall': eval_output[1], \
            'iou_min': eval_output[8], 'iou_max': eval_output[9], \
            'iou_mean': np.mean(eval_output[10]),
            'triangle': '3000',
            'square': '3000',
            'rhombus': '6000',
            'hexagon': '0',
            'circle': '3000',
            'total figures': '15000'
        }, ignore_index=True
    )
    eval_output = eval_cnn(test_2_loader, batch_size=32)
    status = status.append(
        {
            'test': f'{epoch}_test_hexagon', \
            'precision': eval_output[0], 'recall': eval_output[1], \
            'iou_min': eval_output[8], 'iou_max': eval_output[9], \
            'iou_mean': np.mean(eval_output[10]),
            'triangle': '3000',
            'square': '3000',
            'rhombus': '3000',
            'hexagon': '3000',
            'circle': '3000',
            'total figures': '15000'
        }, ignore_index=True
    )
    return status


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
# Протестируем на динамической выборке
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
torch.save(model.state_dict(), 'model_simple_fly.pth')
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history['epoch'], history['train_loss'], label='train_loss')
for i in [0, 1]:
    axes[i].legend()
torch.cuda.empty_cache()
gc.collect()
model = ShapeDetector().to(device)
model.load_state_dict(torch.load('model_simple_fly.pth', map_location=torch.device(device)))
eval_output = eval_cnn(test_1_loader, batch_size=32)
print('precision:', eval_output[0], 'recall:', eval_output[1])
print('iou_min:', eval_output[8], 'iou_max:', eval_output[9])
print('iou_mean:', np.mean(eval_output[10]))
results = pd.DataFrame(
    {
        'test': ['test_1'], \
        'precision': [eval_output[0]], 'recall': [eval_output[1]], \
        'iou_min': [eval_output[8]], 'iou_max': [eval_output[9]], \
        'iou_mean': [np.mean(eval_output[10])],
        'triangle': ['1003'],
        'square': ['988'],
        'rhombus': ['996'],
        'hexagon': ['1023'],
        'circle': ['990'],
        'total figures': ['5000']
    }
)
print(results)
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(np.rollaxis(eval_output[6].numpy(), 0, 3))
plt.subplot(1, 2, 1)
plt.axis("off")
image = Image.new("RGB", (256, 256))
draw = ImageDraw.Draw(image)
shapes = np.array(eval_output[2])
x1, y1 = shapes[0], shapes[1]
x2, y2 = x1 + shapes[2], y1 + shapes[3]
draw.polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], outline='white')
shapes = np.array(eval_output[3])
x1, y1 = shapes[0], shapes[1]
x2, y2 = x1 + shapes[2], y1 + shapes[3]
draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline='red')
plt.imshow(image)
plt.title('iou_min, white - real, red - predicted')

plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(np.rollaxis(eval_output[6].numpy(), 0, 3))
plt.subplot(1, 2, 2)
plt.axis("off")
image = Image.new("RGB", (256, 256))
draw = ImageDraw.Draw(image)
shapes = np.array(eval_output[4])
x1, y1 = shapes[0], shapes[1]
x2, y2 = x1 + shapes[2], y1 + shapes[5]
draw.polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], outline='white')
shapes = np.array(eval_output[5])
x1, y1 = shapes[0], shapes[1]
x2, y2 = x1 + shapes[2], y1 + shapes[3]
draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline='red')
plt.imshow(image)
plt.title('iou_max, white - real, red - predicted')

plt.show()
# Протестируем на статической выборке
drawer.generate_images(5000, 'output', update=False)
train_data = My_Dataset(fly=False, folder='output')
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
torch.cuda.empty_cache()
gc.collect()
model = ShapeDetector().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
history = train_cnn(20, train_loader, batch_size=batch_size)
torch.save(model.state_dict(), 'model_simple_static.pth')
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history['epoch'], history['train_loss'], label='train_loss')
for i in [0, 1]:
    axes[i].legend()
torch.cuda.empty_cache()
gc.collect()
model = ShapeDetector().to(device)
model.load_state_dict(torch.load('model_simple_static.pth', map_location=torch.device(device)))
eval_output = eval_cnn(test_2_loader, batch_size=32)
print('precision:', eval_output[0], 'recall:', eval_output[1])
print('iou_min:', eval_output[8], 'iou_max:', eval_output[9])
print('iou_mean:', np.mean(eval_output[10]))
results = results.append(
    {
        'test': 'test_2', \
        'precision': eval_output[0], 'recall': eval_output[1], \
        'iou_min': eval_output[8], 'iou_max': eval_output[9], \
        'iou_mean': np.mean(eval_output[10]),
        'triangle': '5004',
        'square': '5075',
        'rhombus': '5022',
        'hexagon': '4955',
        'circle': '4944',
        'total figures': '25000'
    }, ignore_index=True
)
print(results)
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(np.rollaxis(eval_output[6].numpy(), 0, 3))
plt.subplot(1, 2, 1)
plt.axis("off")
image = Image.new("RGB", (256, 256))
draw = ImageDraw.Draw(image)
shapes = np.array(eval_output[2])
x1, y1 = shapes[0], shapes[1]
x2, y2 = x1 + shapes[2], y1 + shapes[3]
draw.polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], outline='white')
shapes = np.array(eval_output[3])
x1, y1 = shapes[0], shapes[1]
x2, y2 = x1 + shapes[2], y1 + shapes[3]
draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline='red')
plt.imshow(image)
plt.title('iou_min, white - real, red - predicted')

plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(np.rollaxis(eval_output[6].numpy(), 0, 3))
plt.subplot(1, 2, 2)
plt.axis("off")
image = Image.new("RGB", (256, 256))
draw = ImageDraw.Draw(image)
shapes = np.array(eval_output[4])
x1, y1 = shapes[0], shapes[1]
x2, y2 = x1 + shapes[2], y1 + shapes[5]
draw.polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], outline='white')
shapes = np.array(eval_output[5])
x1, y1 = shapes[0], shapes[1]
x2, y2 = x1 + shapes[2], y1 + shapes[3]
draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline='red')
plt.imshow(image)
plt.title('iou_max, white - real, red - predicted')
plt.show()

# Второй этап тренировок
drawer = PictureDrawer(5, hexagon_status=False, random_status=False)
drawer.generate_images(12000, 'work', update=False)
drawer = PictureDrawer(5, hexagon_status=False, random_status=False)
drawer.generate_images(3000, 'work_test', update=False)
drawer = PictureDrawer(5, hexagon_status=True, random_status=False)
drawer.generate_images(3000, 'work_test_hexagon', update=False)
test_1_data = My_Dataset(folder='work_test', fly=False)
test_2_data = My_Dataset(folder='work_test_hexagon', fly=False)
test_1_loader = torch.utils.data.DataLoader(test_1_data, batch_size=batch_size, shuffle=False)
test_2_loader = torch.utils.data.DataLoader(test_2_data, batch_size=batch_size, shuffle=False)
torch.cuda.empty_cache()
gc.collect()
train_data = My_Dataset(fly=False, folder='work')
test_1_data = My_Dataset(folder='work_test', fly=False)
test_2_data = My_Dataset(folder='work_test_hexagon', fly=False)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_1_loader = torch.utils.data.DataLoader(test_1_data, batch_size=batch_size, shuffle=False)
test_2_loader = torch.utils.data.DataLoader(test_2_data, batch_size=batch_size, shuffle=False)
model = ShapeDetector().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
history = train_cnn(20, train_loader, batch_size=32)
torch.save(model.state_dict(), 'model_simple_start_learning.pth')
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history['epoch'], history['train_loss'], label='train_loss')
for i in [0, 1]:
    axes[i].legend()

# Третий эиап тренировок
for i in range(20):
    drawer.generate_images(400, 'work', update=True)
    results = transfer_learning(i, 20, status=results)
print(results)
