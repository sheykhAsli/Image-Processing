import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import cv2

# بارگذاری مدل VGG16 با وزن‌های پیش‌آموزش داده شده
model = VGG16(weights='imagenet')

def load_and_preprocess_image(image_path):
    # بارگذاری تصویر و تغییر اندازه آن به 224x224 پیکسل
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict(image):
    # پیش‌بینی نوع آفت با استفاده از مدل
    preds = model.predict(image)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return decoded_preds

def draw_predictions(image_path, predictions):
    # بارگذاری تصویر اصلی
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # رسم پیش‌بینی‌ها روی تصویر
    for i, (imagenet_id, label, score) in enumerate(predictions):
        text = f"{label}: {score:.2f}"
        y = 30 + i * 30
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ذخیره تصویر خروجی
    output_path = 'output_with_predictions.jpg'
    cv2.imwrite(output_path, image)
    return output_path

# مسیر تصویر ورودی
image_path = 'path_to_your_image.jpg'

# بارگذاری و پیش‌پردازش تصویر
image = load_and_preprocess_image(image_path)

# پیش‌بینی نوع آفت در تصویر
predictions = predict(image)

# رسم نتایج پیش‌بینی شده روی تصویر
output_path = draw_predictions(image_path, predictions)
print(f"نتایج پیش‌بینی شده در تصویر ذخیره شده در: {output_path}")
