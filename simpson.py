import cv2
import os
import tensorflow as tf
import numpy as np
import random

print(tf.config.list_physical_devices('GPU'))

class Preprocess:

    def __init__(self):
        path_windows = "C:/Users/.../journey-springfield/train/simpsons_dataset"
        path_linux = "/mnt/c/.../journey-springfield/train/simpsons_dataset"
        self.path_train = path_linux
        self.labels = os.listdir(self.path_train)
        self.dataset = []
        self.labels = self.labels[:20] # Редактируем количество классов
        self.abraham = os.listdir(self.path_train + "/" + self.labels[0])
        self.class_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.value_img = 300 # Количество изображений на класс. Ограничение видеопамяти 4Гб, одна картинка 602 Кб
        self.n = 5 # Количество разбиений
        self.step = 100  # Шаг для массива images и labels для загрузки пачками


    def preprocess_image(self, image_path, target_height, target_width, random_=False):
        # Загрузка изображения
        img_raw = tf.io.read_file(image_path)
        img = tf.image.decode_image(img_raw)
        height, width = img.shape[:2]

        # Фильтрация по размеру
        if not (250 < height < 600 and 250 < width < 600):
            return None
        if random_:
            img = tf.image.random_brightness(img, max_delta=0.2) # Случайное изменение яркости
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2) # Случайное изменение контрастности
            img = tf.image.random_hue(img, max_delta=0.02) # Случайное изменение цветов
            img = tf.image.random_saturation(img, lower=0.8, upper=1.2) # Случайное изменение насыщенности
            # img = tf.image.rot90(img, k=random.randint(0, 3)) # Случайное изменение ориентации k = 0: Без вращения.k = 1: Поворот на 90° по часовой стрелке.k = 2: Поворот на 180°.k = 3: Поворот на 270° по часовой стрелке.
            img = tf.image.resize_with_crop_or_pad(img, target_height + 20, target_width + 20) # Случайная обрезка или добавление рамки
            # img = tf.image.random_crop(img, size=[target_height, target_width, 3])
            img = tf.image.random_flip_left_right(img)  # Случайное горизонтальное отражение

        resized_img = tf.image.resize(img, [target_height, target_width])  # Масштабирование изображений
        # resized_img = tf.cast(resized_img, tf.uint8)  # Преобразование к uint8 для отображения cv2
        resized_img = resized_img / 255.0  # Нормализуем изображение

        return resized_img


    def make_dataset_RAM(self):
        # Средние значение ширины и высоты 416 и 409
        target_height, target_width = 224, 224

        for label, value in zip(self.labels, self.dict_classes.values()):

            take_img = os.listdir(self.path_train + "/" + label)
            images = []
            labels = []
            if value < self.value_img:

                while len(labels) != self.value_img:
                    full_path = self.path_train + "/" + label + "/" + random.choice(take_img)
                    processed_image = self.preprocess_image(full_path, target_height, target_width, random_=True)
                    if processed_image is not None:
                        images.append(processed_image)
                        labels.append(self.class_to_index[label])
            else:
                for path in take_img:
                    full_path = self.path_train + "/" + label + "/" + path
                    processed_image = self.preprocess_image(full_path, target_height, target_width)
                    if processed_image is not None:
                        images.append(processed_image)
                        labels.append(self.class_to_index[label])
                    if len(labels) == self.value_img:
                        break

            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
            print(f"Class {label}: Processed {len(images)} images")

            np.save(os.path.join(f"dataset_npy/{label}_images.npy"), images)
            np.save(os.path.join(f"dataset_npy/{label}_labels.npy"), labels)
            print(f"Сохранены данные для класса {label}: {images.shape}, {labels.shape}")


    def make_dataset_ROM(self): # Не эффективно
        # Средние значение ширины и высоты 416 и 409
        target_height, target_width = 224, 224

        for label, value in zip(self.labels, self.dict_classes.values()):

            take_img = os.listdir(self.path_train + "/" + label)
            images = []
            labels = []
            if value < self.value_img:

                while len(labels) != self.value_img:
                    full_path = self.path_train + "/" + label + "/" + random.choice(take_img)
                    processed_image = self.preprocess_image(full_path, target_height, target_width, random_=True)
                    if processed_image is not None:
                        images.append(processed_image)
                        labels.append(self.class_to_index[label])
            else:
                for path in take_img:
                    full_path = self.path_train + "/" + label + "/" + path
                    processed_image = self.preprocess_image(full_path, target_height, target_width)
                    if processed_image is not None:
                        images.append(processed_image)
                        labels.append(self.class_to_index[label])
                    if len(labels) == self.value_img:
                        break

            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
            print(f"Class {label}: Processed {len(images)} images")
            for i in range(self.n):
                start = i * self.step
                stop = (i + 1) * self.step
                np.save(os.path.join(f"dataset_npy_ROM/{label}_{i}_images.npy"), images[start:stop])
                np.save(os.path.join(f"dataset_npy_ROM/{label}_{i}_labels.npy"), labels[start:stop])
                print(f"Сохранены данные для класса {label}: {images[start:stop].shape}, {labels[start:stop].shape}")



    def cv2_view(self):
        label = 'abraham_grampa_simpson'
        for i in range(5):
            img = np.load(f"output_dir/{label}_{i}_images.npy")
            img = np.concatenate(img, axis=0)
            img = (img * 255).astype(np.uint8)

            cv2.imshow("Image", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()



"""{'abraham_grampa_simpson': 812, 'agnes_skinner': 35, 'apu_nahasapeemapetilon': 528, 'barney_gumble': 100, 'bart_simpson': 1131, 'carl_carlson': 71, 'charles_montgomery_burns': 1077, 'chief_wiggum': 798, 'cletus_spuckler': 34, 'comic_book_guy': 376, 'disco_stu': 8, 'edna_krabappel': 376, 'fat_tony': 20, 'gil': 22, 'groundskeeper_willie': 95, 'homer_simpson': 1774, 'kent_brockman': 421, 'krusty_the_clown': 979, 'lenny_leonard': 277, 'lionel_hutz': 3, 'lisa_simpson': 1093, 'maggie_simpson': 90, 'marge_simpson': 1101, 'martin_prince': 61, 'mayor_quimby': 185, 'milhouse_van_houten': 941, 'miss_hoover': 13, 'moe_szyslak': 1188, 'ned_flanders': 1184, 'nelson_muntz': 311, 'otto_mann': 17, 'patty_bouvier': 72, 'principal_skinner': 994, 'professor_john_frink': 44, 'rainier_wolfcastle': 41, 'ralph_wiggum': 66, 'selma_bouvier': 102, 'sideshow_bob': 698, 'sideshow_mel': 30, 'snake_jailbird': 46, 'troy_mcclure': 3, 'waylon_smithers': 147}"""
