from train_model import RAM
from CNN_Model import Model
from simpson import Preprocess
import tensorflow as tf
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

prepro = Preprocess()
# prepro.make_dataset_RAM() # Подгружаем картинки на ПЗУ
# prepro.cv2_view()

train = RAM()
train.train_model()
