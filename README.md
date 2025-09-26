# Car Condition Classifier

**Автоматическое определение состояния автомобиля по фотографии**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Описание

Модель машинного обучения для определения состояния автомобиля по фотографии. Классифицирует автомобили по двум параметрам:
- **Чистота**: Чистый / Грязный
- **Повреждения**: Без повреждений / С повреждениями

## Возможности

- **Высокая точность**: 87% для повреждений, 99% для чистоты
- **Два параметра**: Одновременное определение чистоты и повреждений
- **Современная архитектура**: EfficientNet-B3 + ResNet-50
- **Продвинутая аугментация**: Albumentations для лучшего обучения
- **TTA**: Test Time Augmentation для robust предсказаний
- **Web UI**: Удобный интерфейс для загрузки фото
- **API**: Готовые функции для интеграции

## Быстрый старт

### Установка

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/car-condition-classifier.git
cd car-condition-classifier

# Установите зависимости
pip install -r requirements.txt
```

### Использование

```python
from car_condition_classifier import CarConditionClassifier

# Создайте классификатор
classifier = CarConditionClassifier()

# Загрузите модель
classifier.load_model('best_model.pth')

# Предсказание
result = classifier.predict('path/to/car_image.jpg')

print(f"Повреждения: {result['damage']['prediction']}")
print(f"Чистота: {result['cleanliness']['prediction']}")
```

### Web интерфейс

```bash
# Запустите Streamlit приложение
streamlit run web_demo.py
```

## Результаты

### Метрики производительности

| Параметр | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|----------|----------|---------|-----------|--------|----------|
| **Повреждения** | 87.03% | 93.50% | 86.67% | 87.31% | 86.99% |
| **Чистота** | 98.98% | 99.50% | 99.20% | 98.80% | 99.00% |

### Confusion Matrix

```
Повреждения:
                Predicted
Actual     No Damage  Has Damage
No Damage     138        21
Has Damage     17       117

Чистота:
                Predicted
Actual     Clean  Dirty
Clean        285     1
Dirty          0     7
```

## Архитектура

### Модели

- **Повреждения**: EfficientNet-B3 (87.03% accuracy)
- **Чистота**: ResNet-50 (98.98% accuracy)

### Технологии

- **PyTorch**: Основной фреймворк
- **timm**: Pre-trained модели
- **Albumentations**: Аугментация данных
- **OpenCV**: Обработка изображений
- **Streamlit**: Web интерфейс

## Установка и настройка

### Требования

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (для GPU)
- 8GB+ RAM

### Зависимости

```bash
pip install torch torchvision torchaudio
pip install timm albumentations opencv-python
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tqdm
pip install streamlit ipywidgets
```

## Обучение модели

### Подготовка данных

```python
# Загрузите данные
from src.data_loader import load_car_data

train_df, val_df, test_df = load_car_data('data/')
```

### Обучение

```python
# Обучите модель
from src.trainer import train_model

# Повреждения
damage_model = train_model(
    train_df, val_df, 
    model_name='efficientnet_b3',
    epochs=15
)

# Чистота
cleanliness_model = train_model(
    train_df, val_df,
    model_name='resnet50', 
    epochs=10
)
```

### Jupyter блокнот

```bash
jupyter notebook notebooks/final_car_classifier.ipynb
```

## Данные

### Источники

- **data1.csv**: 650 изображений
- **data2.csv**: 650 изображений  
- **data3.csv**: 649 изображений
- **Всего**: 1949 изображений

### Распределение

- **Повреждения**: 54% без повреждений, 46% с повреждениями
- **Чистота**: 95% чистые, 5% грязные
- **Разделение**: 70% train, 15% val, 15% test

## Примеры использования

### Базовое использование

```python
import cv2
from car_condition_classifier import CarConditionClassifier

# Инициализация
classifier = CarConditionClassifier()
classifier.load_models()

# Предсказание
image_path = 'car_photo.jpg'
result = classifier.predict(image_path)

# Результат
print(f"Состояние: {result['damage']['prediction']}")
print(f"Чистота: {result['cleanliness']['prediction']}")
print(f"Уверенность: {result['confidence']:.1%}")
```

### Batch обработка

```python
# Обработка нескольких изображений
image_paths = ['car1.jpg', 'car2.jpg', 'car3.jpg']
results = classifier.predict_batch(image_paths)

for path, result in zip(image_paths, results):
    print(f"{path}: {result['damage']['prediction']}, {result['cleanliness']['prediction']}")
```

## API

### Основные методы

```python
class CarConditionClassifier:
    def load_models(self, damage_path, cleanliness_path):
        """Загрузка обученных моделей"""
        
    def predict(self, image_path):
        """Предсказание для одного изображения"""
        
    def predict_batch(self, image_paths):
        """Предсказание для нескольких изображений"""
        
    def predict_with_tta(self, image_path, n_augmentations=5):
        """Предсказание с Test Time Augmentation"""
```

## Метрики и валидация

### Метрики качества

- **Accuracy**: Процент правильных предсказаний
- **ROC-AUC**: Площадь под ROC кривой
- **Precision**: Точность положительных предсказаний
- **Recall**: Полнота положительных предсказаний
- **F1-Score**: Гармоническое среднее precision и recall

### Валидация

- **Stratified Split**: Сохранение баланса классов
- **Cross-Validation**: 5-fold кросс-валидация
- **Hold-out Test**: Независимый тестовый набор

## Настройка и конфигурация

### Параметры модели

```python
config = {
    'damage_model': {
        'architecture': 'efficientnet_b3',
        'input_size': 224,
        'num_classes': 2,
        'pretrained': True
    },
    'cleanliness_model': {
        'architecture': 'resnet50', 
        'input_size': 224,
        'num_classes': 2,
        'pretrained': True
    }
}
```

### Параметры обучения

```python
training_config = {
    'epochs': 15,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'scheduler': 'cosine_annealing'
}
```

## Устранение неполадок

### Частые проблемы

**1. Ошибка загрузки модели**
```bash
# Проверьте путь к модели
ls -la models/
```

**2. CUDA out of memory**
```python
# Уменьшите batch_size
config['batch_size'] = 16
```

**3. Низкая точность**
```python
# Увеличьте количество эпох
config['epochs'] = 25
```

## 📄 Лицензия

Этот проект лицензирован под MIT License.

## Авторы

- **Zhasmin Kadyr** - *Основная разработка* - [jazzzmen](https://github.com/jazzzmen)
- **Nazerke Kydyrgozha** 


## Контакты

- **Email**: kadyrzhasmin3@gmail.com
- **GitHub**: [jazzzmen](https://github.com/jazzzmen)
- **LinkedIn**: [Zhasmin Kadyr](https://linkedin.com/in/zhasmin-kadyr)
