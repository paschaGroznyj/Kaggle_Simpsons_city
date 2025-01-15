import os
import tensorflow as tf
import numpy as np
from tf_keras.models import load_model # Для Windows
# from tensorflow.keras.models import load_model
import csv

# Подготовленный словарь для преобразования argmax в текстовую метку
rename_classes = {0: 'abraham_grampa_simpson', 1: 'agnes_skinner', 2: 'apu_nahasapeemapetilon', 3: 'barney_gumble', 4: 'bart_simpson', 5: 'carl_carlson', 6: 'charles_montgomery_burns', 7: 'chief_wiggum', 8: 'cletus_spuckler', 9: 'comic_book_guy', 10: 'disco_stu', 11: 'edna_krabappel', 12: 'fat_tony', 13: 'gil', 14: 'groundskeeper_willie', 15: 'homer_simpson', 16: 'kent_brockman', 17: 'krusty_the_clown', 18: 'lenny_leonard', 19: 'lionel_hutz', 20: 'lisa_simpson', 21: 'maggie_simpson', 22: 'marge_simpson', 23: 'martin_prince', 24: 'mayor_quimby', 25: 'milhouse_van_houten', 26: 'miss_hoover', 27: 'moe_szyslak', 28: 'ned_flanders', 29: 'nelson_muntz', 30: 'otto_mann', 31: 'patty_bouvier', 32: 'principal_skinner', 33: 'professor_john_frink', 34: 'rainier_wolfcastle', 35: 'ralph_wiggum', 36: 'selma_bouvier', 37: 'sideshow_bob', 38: 'sideshow_mel', 39: 'snake_jailbird', 40: 'troy_mcclure', 41: 'waylon_smithers'}
# path = "/mnt/c/.../journey-springfield/testset/testset"
path = "C:/.../journey-springfield/testset/testset"
name_dict = {name: 1 for name in os.listdir(path)}
model = load_model('keras_models/best_city_simpsons_42_300_image.keras')
def preprocess_image(image_path, target_height, target_width):
    # Загрузка изображения
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_raw)
    resized_img = tf.image.resize(img, [target_height, target_width])  # Масштабирование изображений
    resized_img = resized_img / 255.0  # Нормализуем изображение

    return resized_img

images = []
names = []
# Формируем массив тестовых изображений
for img in name_dict.keys():
    full_path = f'{path}/{img}'
    processed_image = preprocess_image(full_path, 224, 224)
    images.append(processed_image)
    names.append(img)

images = np.array(images, dtype=np.float16)
print(f"Processed {len(images)} images")

# Предсказываем класс изображений
predictions = model.predict(images)

csv_dict = {}
for name, prediction in zip(names, predictions):
    class_ = np.argmax(prediction)
    label = rename_classes[class_]
    csv_dict[name] = label

# Загружаем в итоговый csv
with open('result_test.csv', mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Id", "Expected"])
    for key, value in csv_dict.items():
        writer.writerow([key, value])

