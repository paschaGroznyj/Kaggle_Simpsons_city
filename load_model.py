from tensorflow.keras.models import load_model
from train_model import RAM
from CNN_Model import Model


parameters = Model()
data = RAM()
X_test, y_test = data.data_preparation_for_test()
print("#############################################Загрузили\n")
model = load_model('best_city_simpsons.keras')
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=parameters.batch_size, verbose=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")