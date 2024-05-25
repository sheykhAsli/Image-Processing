import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing import image

# مسیر داده‌ها
train_data_dir = 'path_to_train_data'  # مسیر داده‌های آموزشی
validation_data_dir = 'path_to_validation_data'  # مسیر داده‌های اعتبارسنجی

# پارامترهای اصلی
img_width, img_height = 224, 224
batch_size = 32
epochs = 25
num_classes = 10  # تعداد کلاس‌ها (آفات)

# آماده‌سازی داده‌ها
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# بارگذاری مدل VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# افزودن لایه‌های جدید به مدل
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# ساخت مدل نهایی
model = Model(inputs=base_model.input, outputs=predictions)

# فریز کردن لایه‌های اولیه VGG16
for layer in base_model.layers:
    layer.trainable = False

# کامپایل کردن مدل
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# آموزش مدل
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator)

# ذخیره مدل آموزش دیده
model.save('plant_pest_detector_vgg16.h5')

# تابع پیش‌بینی
def predict_pest(img_path):
    model = load_model('plant_pest_detector_vgg16.h5')
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', np.argmax(preds[0]))

# نمونه فراخوانی تابع پیش‌بینی
predict_pest('path_to_test_image.jpg')