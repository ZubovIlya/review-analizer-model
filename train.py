import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from torchmetrics.functional import precision, recall, f1_score
import json
import os


# Определение класса для пользовательского Dataset
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=512):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        # Токенизация текста
        tokens = self.tokenizer(
            review, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': label
        }


# Функция для получения данных и разделения на train/test
def get_train_test(path_json, nclass=1):
    with open(path_json, 'r') as file:
        reviews = json.load(file)
    reviews = list(reviews.values())
    y_reviews = [nclass] * len(reviews)

    return train_test_split(reviews, y_reviews, test_size=0.2, random_state=42)


# Функция для усредненного пулинга скрытых состояний
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Модель нейронной сети
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(384, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


# # Функция для обучения модели
# def train(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0

#     for batch in dataloader:
#         inputs = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs, attention_mask)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# Функция для обучения модели
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Функция для инференса (оценки модели)
def inference(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)

    return total_loss / len(dataloader), all_outputs, all_labels

def get_dataset(path_usef_neg, path_usef_pos, path_usel_neg, path_usel_pos, device):

    # Инициализация токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    transformer_model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

    # Получаем данные для обучения и тестирования
    x_train_usef_n, x_test_usef_n, y_train_usef_n, y_test_usef_n = get_train_test(path_usef_neg, nclass=1)
    x_train_usef_p, x_test_usef_p, y_train_usef_p, y_test_usef_p = get_train_test(path_usef_pos, nclass=2)
    x_train_usel_n, x_test_usel_n, y_train_usel_n, y_test_usel_n = get_train_test(path_usel_neg, nclass=3)
    x_train_usel_p, x_test_usel_p, y_train_usel_p, y_test_usel_p = get_train_test(path_usel_pos, nclass=4)

    x_train = x_train_usef_n + x_train_usef_p + x_train_usel_n + x_train_usel_p
    y_train = torch.zeros((len(x_train), 4))
    for i, y in enumerate(y_train_usef_n + y_train_usef_p + y_train_usel_n + y_train_usel_p):
        y_train[i, y - 1] = 1

    x_test = x_test_usef_n + x_test_usef_p + x_test_usel_n + x_test_usel_p
    y_test = torch.zeros((len(x_test), 4))
    for i, y in enumerate(y_test_usef_n + y_test_usef_p + y_test_usel_n + y_test_usel_p):
        y_test[i, y - 1] = 1

    batch_x_train = tokenizer(x_train, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_x_test = tokenizer(x_test, max_length=512, padding=True, truncation=True, return_tensors='pt')

    x_train = transformer_model(**batch_x_train)
    x_test = transformer_model(**batch_x_test)

    x_train = average_pool(x_train.last_hidden_state, batch_x_train['attention_mask']).detach()
    x_test = average_pool(x_test.last_hidden_state, batch_x_test['attention_mask']).detach()

    return x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)


# Главная функция
def main(path_usef_neg, path_usef_pos, path_usel_neg, path_usel_pos, path_checkpoint="./checkpoints/"):

    os.makedirs(path_checkpoint, exist_ok=True)

    # Инициализация модели, функции потерь и оптимизатора
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    x_train, y_train, x_test, y_test = get_dataset(path_usef_neg, path_usef_pos, path_usel_neg, path_usel_pos, device)

    epochs = 1000
    best_f1 = 0
    for epoch in range(epochs):
        net.train()
        loss_accum = []
    
        optimizer.zero_grad()
        outputs = net(x_train)  # Forward pass
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
        net.eval()
        outputs = net(x_test)  # Forward pass
        loss = criterion(outputs, y_test)
        pres = precision(outputs, y_test, 'binary').item()
        rec = recall(outputs, y_test, 'binary').item()
        f1 = f1_score(outputs, y_test, 'binary').item()
    
        if f1 > 0 and f1 > best_f1:
            best_f1 = f1
            torch.save(net.state_dict(), f"{path_checkpoint}model_best_{round(best_f1, 4)}.pt")
    
        print(epoch, loss.item(), pres, rec, f1)

    print(f"Best F1 Score: {best_f1}")

# Запуск главной функции
if __name__ == "__main__":

    # Путь к JSON-файлам
    path_usef_neg = './Data/useful_negative_reviews.json'
    path_usef_pos = './Data/useful_positive_reviews.json'
    path_usel_neg = './Data/useless_negative_reviews.json'
    path_usel_pos = './Data/useless_positive_reviews.json'

    main(path_usef_neg, path_usef_pos, path_usel_neg, path_usel_pos)
