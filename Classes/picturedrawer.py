from PIL import Image, ImageDraw
import random
import math
import json
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PictureDrawer:
    def __init__(
            self,
            num_rectangles,
            hexagon_status=False,
            random_status=True
    ):
        '''
        На входе:
        num_rectangles - количество фигур на картинке
        hexagon_status=False - набор фигур не будет иметь фигуры гексагон
        и наоборот
        random_status=True - набор фигур будет рандомным и наоборот
        '''
        self.hexagon_status = hexagon_status
        self.num_rectangles = num_rectangles
        self.random_status = random_status
        self.angle = random.randint(0, 359)
        # Если нам нужен гексагон на картинке, будем генерить его
        if hexagon_status:
            self.figures = [
                'triangle',
                'circle',
                'square',
                'rhombus',
                'hexagon'
            ]
        else:
            self.figures = ['triangle', 'circle', 'square', 'rhombus']

    def generate_random_color(self):
        # Гененрируем рандомный цвет
        # На выходе: рандомные r, g, b
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r, g, b)

    def generate_shapes(self):
        # Генерим области для последующего построения фигур
        self.rectangles = []
        self.shapes = []
        for _ in range(self.num_rectangles):
            self.generate_rectangle()

    def generate_rectangle(self):
        # Генерим прямоугольники, которые однозначно не пересекаются
        # На выходе: список прямоугольных областей
        while True:
            x1 = random.randint(0, 256 - 25)
            y1 = random.randint(0, 256 - 25)
            width = random.randint(25, min(150, 256 - x1))
            height = random.randint(25, min(150, 256 - y1))
            x2 = x1 + width
            y2 = y1 + height
            new_rectangle = (x1, y1, x2, y2)

            overlap = False
            for rectangle in self.rectangles:
                if self.check_overlap(new_rectangle, rectangle):
                    overlap = True
                    break

            if not overlap:
                self.rectangles.append(new_rectangle)
                break

    def check_overlap(self, rect1, rect2):
        # Проверка на пересечение новой прямоугольной области с уже сделанными
        # ранее
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        if x1 > x4 or x2 < x3 or y1 > y4 or y2 < y3:
            return False
        return True

    def draw_triangle(self, rectangle, new_color):
        # Рисуем треугольник
        x1, y1, x2, y2 = rectangle
        x3 = random.randint(x1, x2)
        y3 = random.randint(y1, y2)
        triangle = (x1, y1, x2, y2, x3, y3)
        self.draw.polygon(triangle, new_color)
        height = y2 - y1
        width = x2 - x1
        return x1, y1, height, width

    def draw_circle(self, rectangle, new_color):
        # Рисуем круг
        x1, y1, x2, y2 = rectangle
        center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
        radius = min((x2 - x1) / 2, (y2 - y1) / 2)
        points = []
        for i in range(360):
            x = center[0] + int(radius * math.cos(math.radians(i)))
            y = center[1] + int(radius * math.sin(math.radians(i)))
            points.append((x, y))
        self.draw.polygon(points, new_color)
        return center[0] - radius, center[1] - radius, radius * 2, radius * 2

    def draw_square(self, rectangle, new_color):
        # Рисуем квадрат
        x1, y1, x2, y2 = rectangle
        center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
        radius = min((x2 - x1) / 2, (y2 - y1) / 2)
        angle_rad = math.radians(self.angle)
        while True:
            radius += 1
            points = [
                (center[0], center[1] - radius),
                (center[0] + radius, center[1]),
                (center[0], center[1] + radius),
                (center[0] - radius, center[1])
            ]

            # Вращение по центру
            rotated_points = []
            for point in points:
                x_rot = center[0] + math.cos(angle_rad) * (point[0] - center[0]) - math.sin(angle_rad) * (
                        point[1] - center[1])
                y_rot = center[1] + math.sin(angle_rad) * (point[0] - center[0]) + math.cos(angle_rad) * (
                        point[1] - center[1])
                rotated_points.append((x_rot, y_rot))
            x_min = min([rotated_points[i][0] for i in range(4)])
            x_max = max([rotated_points[i][0] for i in range(4)])
            y_min = min([rotated_points[i][1] for i in range(4)])
            y_max = max([rotated_points[i][1] for i in range(4)])
            height = y_max - y_min
            width = x_max - x_min
            # Проверка на требование к размерам описывающего прямоугольника
            flag = False
            if height < 25 or width < 25:
                flag = True
            if not flag:
                self.draw.polygon(rotated_points, new_color)
                return x_min, y_min, height, width

    def draw_rhombus(self, rectangle, new_color):
        # Рисуем ромб
        x1, y1, x2, y2 = rectangle
        center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
        radius = int(min((x2 - x1) / 2, (y2 - y1) / 2))
        while True:
            radius += 1
            side = random.randint(1, radius)
            angle_rad = math.radians(self.angle)
            points = [
                (center[0], center[1] - radius),
                (center[0] + side, center[1]),
                (center[0], center[1] + radius),
                (center[0] - side, center[1])
            ]

            # Вращение по центру
            rotated_points = []
            for point in points:
                x_rot = center[0] + math.cos(angle_rad) * (point[0] - center[0]) - math.sin(angle_rad) * (
                        point[1] - center[1])
                y_rot = center[1] + math.sin(angle_rad) * (point[0] - center[0]) + math.cos(angle_rad) * (
                        point[1] - center[1])
                rotated_points.append((x_rot, y_rot))
            x_min = min([rotated_points[i][0] for i in range(4)])
            x_max = max([rotated_points[i][0] for i in range(4)])
            y_min = min([rotated_points[i][1] for i in range(4)])
            y_max = max([rotated_points[i][1] for i in range(4)])
            height = y_max - y_min
            width = x_max - x_min
            # Проверка на требование к размерам описывающего прямоугольника
            flag = False
            if height < 25 or width < 25:
                flag = True
            if not flag:
                self.draw.polygon(rotated_points, new_color)
                return x_min, y_min, height, width

    def draw_hexagon(self, rectangle, new_color):
        # Рисуем гексагон
        x1, y1, x2, y2 = rectangle
        center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
        radius = int(min((x2 - x1) / 2, (y2 - y1) / 2))
        while True:
            radius += 1
            side = random.randint(1, radius)
            rotate_angle_rad = math.radians(self.angle)
            vertices = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.pi / 180 * angle_deg
                x = center[0] + radius * math.cos(angle_rad)
                y = center[1] + radius * math.sin(angle_rad)
                vertices.append((x, y))
            # Вращение по центру
            rotated_points = []
            for point in vertices:
                x_rot = center[0] \
                        + math.cos(rotate_angle_rad) * (point[0] - center[0]) \
                        - math.sin(rotate_angle_rad) * (point[1] - center[1])
                y_rot = center[1] \
                        + math.sin(rotate_angle_rad) * (point[0] - center[0]) \
                        + math.cos(rotate_angle_rad) * (point[1] - center[1])
                rotated_points.append((x_rot, y_rot))
            x_min = min([rotated_points[i][0] for i in range(6)])
            x_max = max([rotated_points[i][0] for i in range(6)])
            y_min = min([rotated_points[i][1] for i in range(6)])
            y_max = max([rotated_points[i][1] for i in range(6)])
            height = y_max - y_min
            width = x_max - x_min
            # Проверка на требование к размерам описывающего прямоугольника
            flag = False
            if height < 25 or width < 25:
                flag = True
            if not flag:
                self.draw.polygon(rotated_points, new_color)
                return x_min, y_min, height, width

    def draw_figure(self, rectangle, figure, new_color):
        # Добавляем те фигуры, которые приняи на вход класса
        if figure == 'triangle':
            x1, y1, height, width = self.draw_triangle(rectangle, new_color)
        elif figure == 'circle':
            x1, y1, height, width = self.draw_circle(rectangle, new_color)
        elif figure == "square":
            x1, y1, height, width = self.draw_square(rectangle, new_color)
        elif figure == "hexagon":
            x1, y1, height, width = self.draw_hexagon(rectangle, new_color)
        elif figure == "rhombus":
            x1, y1, height, width = self.draw_rhombus(rectangle, new_color)
        self.shapes.append({
            "id": len(self.shapes) + 1,
            "name": figure,
            'region': {
                'origin': {"x": x1, "y": y1},
                'size': {"width": width, "height": height}
            },
        })

    def draw_rectangles(self):
        # Рисуем фигуры в зависимости от необходимости рисовать рандомно
        # либо с/без гексагона
        background_color = self.generate_random_color()
        self.image = Image.new("RGB", (256, 256), background_color)
        self.draw = ImageDraw.Draw(self.image)
        num_figures = 0
        for rectangle in self.rectangles:
            new_color = self.generate_random_color()
            if background_color == new_color:
                new_color = self.generate_random_color()
            else:
                if self.random_status:
                    figure = random.choice(self.figures)
                else:
                    figure = self.figures[num_figures]
                # self.draw.rectangle(rectangle)
                self.draw_figure(rectangle, figure, new_color)
                if num_figures >= len(self.figures):
                    num_figures = 0
                elif num_figures < len(self.figures) - 1:
                    num_figures += 1

        return self.image, self.shapes, self.figures

    def generate_images(self, num_images, output_folder, update=False):
        '''
        Гененрим фигуры, сохраняем их в файл и сохраняем в файл характтеристики
        описываемых их областей
        '''
        # Если каталога нет, то делаем
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Читаем файлы в каталоге
        files = os.listdir(output_folder)

        # Если файлов нет или мы не продолжаем запись, то начинаем с начала
        if not files or not update:
            last_num = 0
        elif not update:
            last_num = 0
        # Если каталог не пуст и мы продолжаем добавлять файлы, то продолжаем
        # нумерацию
        else:
            # Сортируем файлы
            files.sort()
            # Берем последний
            last_file = files[-1]
            # Берем номер последнего файла
            last_num = int(last_file.split('.')[0])

        # data = json.loads(data_json)
        if 'data_json.json' in files and update:
            data_filename = os.path.join(output_folder, 'data_json.json')
            data = json.loads(data_filename)
        else:
            data = {}
            data_filename = os.path.join(output_folder, 'data_json.json')

        for i in range(num_images):
            self.generate_shapes()
            image, shapes, figures = self.draw_rectangles()
            image_filename = os.path.join(output_folder, f'{last_num + i + 1:05}.png')
            image.save(image_filename)
            shape_filename = os.path.join(output_folder, f'{last_num + i + 1:05}.json')
            data[last_num + i + 1] = {'image': f'{last_num + i + 1:05}.png', 'file': f'{last_num + i + 1:05}.json'}
            with open(shape_filename, 'w') as f:
                json.dump(shapes, f)
            with open(data_filename, 'w') as f:
                json.dump(data, f)


class My_Dataset(Dataset):
    """Класс генерации torch Dataset"""

    def __init__(self, transform=False, aug=None, fly=False, folder=False):
        self.fly = fly
        self.folder = folder
        if not self.fly:
            with open(os.path.join(self.folder, 'data_json.json'), 'r') as f:
                data = json.load(f)
            print('Количество файлов', ':', len(data))
            self.data_dict = data
        else:
            print('Количество картинок в обучении', ':', 1000)
        if self.folder:
            with open(os.path.join(self.folder, 'data_json.json'), 'r') as f:
                data = json.load(f)
            print('Количество файлов', ':', len(data))
            self.data_dict = data
        self.transform = transform
        self.aug = aug

    def __len__(self):
        if self.fly:
            return 1000
        else:
            return len(self.data_dict)

    def __getitem__(self, idx):
        if not self.transform:
            self.transform = transforms.Compose([
                # you may add anything, e.g. augmentation
                # transforms.CenterCrop(size=120),
                # transforms.Resize(size=(256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        if not self.fly:
            index = str(idx + 1)
            sample_dict = self.data_dict[index]
            img_name = sample_dict['image']
            file_dict = sample_dict['file']

            with open(os.path.join(self.folder, file_dict), 'r') as f:
                data = json.load(f)
                labels = []
                names = []
                for i in range(5):
                    names.append(data[i]['name'])
                    labels.append([
                        data[i]['region']['origin']['x'],
                        data[i]['region']['origin']['y'],
                        data[i]['region']['size']['width'],
                        data[i]['region']['size']['height']
                    ])
            label = torch.LongTensor(labels)
            image = Image.open(os.path.join('output', img_name))
        else:
            drawer.generate_shapes()
            image, shapes, _ = drawer.draw_rectangles()
            labels = []
            names = []
            for i in range(5):
                names.append(shapes[i]['name'])
                labels.append([
                    shapes[i]['region']['origin']['x'],
                    shapes[i]['region']['origin']['y'],
                    shapes[i]['region']['size']['width'],
                    shapes[i]['region']['size']['height']
                ])
            label = torch.LongTensor(labels)
        if self.aug:
            sample = self.aug(
                image=image,
            )
        else:
            if not self.fly:
                sample = {
                    'image': image,
                    'label': label,
                    'image_name': img_name,
                    'file_name': file_dict,
                    'names': names
                }
            else:
                sample = {
                    'image': image,
                    'label': label,
                    'image_name': 'fly',
                    'file_name': 'fly',
                    'names': names
                }
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool0 = nn.MaxPool2d(kernel_size=2, return_indices=True)  # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)  # 128 -> 64
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)  # 64 -> 32
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)  # 32 -> 16
        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(kernel_size=2)  # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.upsample1 = nn.MaxUnpool2d(kernel_size=2)  # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upsample2 = nn.MaxUnpool2d(kernel_size=2)  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.upsample3 = nn.MaxUnpool2d(kernel_size=2)  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
        )
        self.fc1 = nn.Flatten()
        self.fc2 = nn.Linear(256 * 256, 20)

    def forward(self, x):
        # encoder
        enc0 = self.enc_conv0(x)
        pool0, indices_e0 = self.pool0(enc0)
        enc1 = self.enc_conv1(pool0)
        pool1, indices_e1 = self.pool1(enc1)
        enc2 = self.enc_conv2(pool1)
        pool2, indices_e2 = self.pool2(enc2)
        enc3 = self.enc_conv3(pool2)
        pool3, indices_e3 = self.pool3(enc3)
        # bottleneck
        b = self.bottleneck_conv(pool3)
        # decoder
        d0 = self.dec_conv0(torch.cat((self.upsample0(b, indices_e3), enc3), dim=1))
        d1 = self.dec_conv1(torch.cat((self.upsample1(d0, indices_e2), enc2), dim=1))
        d2 = self.dec_conv2(torch.cat((self.upsample2(d1, indices_e1), enc1), dim=1))
        d3 = self.dec_conv3(torch.cat((self.upsample3(d2, indices_e0), enc0), dim=1))  # no activation
        d4 = self.fc1(d3)
        d5 = self.fc2(d4)
        d6 = d5.view(-1, 5, 4)
        return d6


def main():
    drawer = PictureDrawer(5, hexagon_status=True, random_status=False)
    # drawer = PictureDrawer(5, hexagon_status=False, random_status=False)
    # drawer = PictureDrawer(5, hexagon_status=True, random_status=True)
    # drawer = PictureDrawer(5, hexagon_status=False, random_status=True)
    drawer.generate_images(100, 'first_stage', update=False)
    # drawer.generate_images(100, 'output', update = True)


if __name__ == "__main__":
    main()
