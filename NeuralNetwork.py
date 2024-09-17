import os  # Работа с файловой системой
import cv2  # OpenCV для работы с изображениями
import numpy as np  # Библиотека для работы с массивами
import tensorflow as tf  # TensorFlow для нейронных сетей
from tensorflow.keras import layers, models  # Keras слои и модели из TensorFlow
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Аугментация изображений
from tensorflow.keras.preprocessing import image  # Работа с изображениями
import matplotlib.pyplot as plt  # Визуализация данных

# ============== 1. Предобработка данных ==============

# Функция для предобработки изображений
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Нормализация
    return img

# Пример обработки одного изображения
input_dir = 'data/train/defects/'
output_dir = 'data/preprocessed/defects/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for img_file in os.listdir(input_dir):
    img = preprocess_image(os.path.join(input_dir, img_file))
    cv2.imwrite(os.path.join(output_dir, img_file), img * 255)

# ============== 2. Создание модели нейросети ==============

# Построение CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Бинарная классификация (дефект/норма)
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ============== 3. Загрузка и аугментация данных ==============

# Аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True)

# Тестовые данные - только нормализация
test_datagen = ImageDataGenerator(rescale=1./255)

# Загрузка данных из директории
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# ============== 4. Обучение модели ==============

# Обучение модели
batch_size = 32
steps_per_epoch = 769 // batch_size

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

#model.fit(
    #train_generator,
    #steps_per_epoch=461,
    #epochs=10,
    #validation_data=validation_generator,
    #validation_steps=50)

# ============== 5. Оценка модели на тестовых данных ==============

# Оценка модели
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy:.2f}')

# ============== 6. Сохранение модели ==============

# Сохранение модели
model.save('construction_defect_model.h5')

# Загрузка модели для использования в будущем
new_model = tf.keras.models.load_model('construction_defect_model.h5')

# ============== 7. Пример использования модели для предсказания ==============

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = new_model.predict(img_array)

    if prediction[0] > 0.5:
        print(f'{img_path}: Defect detected')
    else:
        print(f'{img_path}: No defect detected')

# Тестирование на новом изображении
predict_image('data/test/defects/sample.jpg')

# ============== 8. Визуализация изображений и предсказаний ==============

def plot_image_and_prediction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = new_model.predict(img_array)

    plt.imshow(img)
    plt.axis('off')

    if prediction[0] > 0.5:
        plt.title('Defect detected')
    else:
        plt.title('No defect detected')

    plt.show()

# Визуализация изображения с предсказанием
plot_image_and_prediction('data/test/defects/sample.jpg')
