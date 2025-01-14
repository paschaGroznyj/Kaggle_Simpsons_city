# Обучение сверточной сети на Tensorflow

## Задание

было взято на портале stepik от университета МИФИ, перенаправлено на Kaggle

ссылка на курс stepik: https://stepik.org/lesson/345648/step/1?unit=824917

ссылка на задание и датасет: https://www.kaggle.com/competitions/journey-springfield

**Цель**: обучить нейронную сеть различать 42-х жителей города

## Проблемы при решении задания

1. **Дисбаланс классов**. Буквально, в одном классе больше 1000 изображений, в другом 3
вот полный список: {'abraham_grampa_simpson': 913, 'agnes_skinner': 42, 'apu_nahasapeemapetilon': 623, 'barney_gumble': 106, 'bart_simpson': 1342, 'carl_carlson': 98, 'charles_montgomery_burns': 1193, 'chief_wiggum': 986, 'cletus_spuckler': 47, 'comic_book_guy': 469, 'disco_stu': 8, 'edna_krabappel': 457, 'fat_tony': 27, 'gil': 27, 'groundskeeper_willie': 121, 'homer_simpson': 2246, 'kent_brockman': 498, 'krusty_the_clown': 1206, 'lenny_leonard': 310, 'lionel_hutz': 3, 'lisa_simpson': 1354, 'maggie_simpson': 128, 'marge_simpson': 1291, 'martin_prince': 71, 'mayor_quimby': 246, 'milhouse_van_houten': 1079, 'miss_hoover': 17, 'moe_szyslak': 1452, 'ned_flanders': 1454, 'nelson_muntz': 358, 'otto_mann': 32, 'patty_bouvier': 72, 'principal_skinner': 1194, 'professor_john_frink': 65, 'rainier_wolfcastle': 45, 'ralph_wiggum': 89, 'selma_bouvier': 103, 'sideshow_bob': 877, 'sideshow_mel': 40, 'snake_jailbird': 55, 'troy_mcclure': 8, 'waylon_smithers': 181}

2. **Дисбаланс размеров изображений**. Средний размер изображений: 416/409. Максимальный размер изображения: 1072/1912

3. **Скорость обучения**. Обучать такой набор на ЦПУ отдельная проблема, решим
4. **Нехватка памяти**. Нет доступа к Google Colab, 4 Гб видеопамяти маловато

## Решение проблем

1. **Проведение аугментации изображений**. Синтетически дополняем классы изображениями, если в классе изображений ниже порогового.
Используем методы tensorflow
2. **Отбрасываем слишком большие или слишком малые изображения**.
3. **Ставим Linux консоль**. Скачиваем Ubuntu, для Nvidia скачиваем Cuda на Linux (поддержка tensorflow на Windows остановлена). Ощущаем прирост скорости обучения
4. **Даем tensorflow всю память видеокарты**. Скриптом снимаем ограничения на потребление видеопамяти. Для идеального эффекта обучения, нужно иметь 500 изображений на класс, в моем случае это было 155, больше просто не помещалось, точность 0.82

## Результаты

1. Структура нейронной сети
    
    ![Image 1](presentation/img.png)
2. Обучение на 20 классах, 300 изображений на класс, батч = 20, точность/правильность обучения 0.92, после 8 эпохи переобучение
![Image 2](presentation/training_plot_20_300_batch_20.png)
3. Обучение на 42 классах, 155 изображений на класс, батч = 10, точность 0.78, не хватает данных для обучения
![Image 2](presentation/training_plot_42_155_batch_10.png)
4. Обучение на 42 классах, 155 изображений на класс, батч = 25, точность 0.8, не хватает данных для обучения
![Image 2](presentation/training_plot_42_155_batch_25.png)
5. Обучение на 42 классах, 155 изображений на класс, батч = 25, точность 0.82, добавлен сверточный слой на 512, стабильно не хватает данных для обучения
![Image 2](presentation/training_plot_42_155_batch_25.png)

### Итог: нужно минимум 500 изображений на класс
