import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# Функция для вычисления MSE (среднеквадратичной ошибки)
def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def visualize_differences(image1, image2):
    # Преобразуем изображения в grayscale, если они еще не в grayscale
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Вычисляем SSIM и маску различий
    score, diff = ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")

    # Пороговая обработка для выделения различий
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Находим контуры различий
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры на оригинальном изображении
    output = image1.copy()
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)

    return output


uploaded_files_container = st.columns([1, 1])
result_container = st.columns([1, 1])
image_gray_array = []

first_uploaded_file = uploaded_files_container[0].file_uploader(
    "Загрузите первое изображение",
    
    key="first_uploader",
    accept_multiple_files=False
)
second_uploaded_file = uploaded_files_container[1].file_uploader(
    "Загрузите второе изображение",
    
    key="second_uploader",
    accept_multiple_files=False
)


if first_uploaded_file:
    first_uploaded_file_name = first_uploaded_file.name.lower()
    
    if first_uploaded_file_name.endswith((".jpg", ".jpeg", ".png")):
        first_file_bytes = np.asarray(bytearray(first_uploaded_file.read()), dtype=np.uint8)
        first_image = cv2.imdecode(first_file_bytes, 1)
        
        first_gray_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        image_gray_array.append(first_gray_image)
        
        result_container[0].image(
            first_gray_image,
            caption="Первое черно-белое изображение",
            use_container_width=True
        )
    else:
        result_container[0].error('Ошибка, неверный формат файла')
        
if second_uploaded_file:
    second_uploaded_file_name = second_uploaded_file.name.lower()
    
    if second_uploaded_file_name.endswith((".jpg", ".jpeg", ".png")):
        second_file_bytes = np.asarray(bytearray(second_uploaded_file.read()), dtype=np.uint8)
        second_image = cv2.imdecode(second_file_bytes, 1)
        
        second_gray_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
        image_gray_array.append(second_gray_image)
        
        result_container[1].image(
            second_gray_image,
            caption="Второе черно-белое изображение",
            use_container_width=True
        )
    else:
        result_container[1].error('Ошибка, неверный формат файла')
        
if len(image_gray_array) == 2:
    # Приведение изображений к одному размеру
    height = min(image_gray_array[0].shape[0], image_gray_array[1].shape[0])
    width = min(image_gray_array[0].shape[1], image_gray_array[1].shape[1])
    
    # Изменение размера изображений
    resized_image1 = cv2.resize(image_gray_array[0], (width, height))
    resized_image2 = cv2.resize(image_gray_array[1], (width, height))
    
    # Вычисляем MSE и SSIM
    mse_value = mse(resized_image1, resized_image2)
    ssim_value, _ = ssim(resized_image1, resized_image2, full=True)
    
    # Порог сходства (можно настроить)
    similarity_threshold = 0.95  # Например, 95% сходства
    
    if ssim_value >= similarity_threshold:
        st.write(f"Изображения похожи (SSIM: {ssim_value:.2f})")
    else:
        st.write(f"Изображения разные (SSIM: {ssim_value:.2f})")
        
        # Визуализация различий
    diff_image = visualize_differences(resized_image1, resized_image2)
    st.image(diff_image, caption="Визуализация различий", use_container_width=True)
    