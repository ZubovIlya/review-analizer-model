import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import json
from train import ReviewDataset, Net  # импортируем нужные классы из вашего основного скрипта

# Загрузка токенизатора и модели трансформера
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
transformer_model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

# Функция для загрузки данных
def load_test_data():
    with open('useless_positive_reviews.json', 'r') as f:
        useless_pos_reviews = json.load(f).values()

    with open('useful_negative_reviews.json', 'r') as f:
        useful_neg_reviews = json.load(f).values()

    # Используем метки классов, аналогичные обучению
    x_test = list(useless_pos_reviews) + list(useful_neg_reviews)
    y_test = [4] * len(useless_pos_reviews) + [1] * len(useful_neg_reviews)

    y_test_one_hot = torch.zeros((len(y_test), 4))
    for i, label in enumerate(y_test):
        y_test_one_hot[i, label - 1] = 1
    
    return x_test, y_test_one_hot


# Загрузка обученной модели
def load_model(model_path='model_best.pt'):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Функция для демонстрации работы модели
def demo(model, tokenizer):
    x_test, y_test = load_test_data()

    # Создание DataLoader для тестовых данных
    test_dataset = ReviewDataset(x_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Прогнозы модели
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            # Прогон через модель
            outputs = model(inputs)
            all_outputs.append(outputs.argmax(dim=1))
            all_labels.append(labels.argmax(dim=1))

    # Объединение всех данных
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Вывод отчета классификации
    report = classification_report(all_labels, all_outputs, target_names=['Useful Negative', 'Useful Positive', 'Useless Negative', 'Useless Positive'])
    print(report)

# Основная функция для демонстрации
def main():
    model = load_model('model_best.pt')
    demo(model, tokenizer)

if __name__ == "__main__":
    main()
