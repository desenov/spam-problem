import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Классификация почтовых отправлений на "спам" и "не спам" на основе их текстового содержания


def generate_data():
    data = {
        "text": [],
        "label": []
    }
    
    for _ in range(200):
        text = " ".join([random.choice(["buy", "get", "free", "money", "now", "won", "lottery", "prize"]) for _ in range(random.randint(5, 15))])
        label = random.choice(["spam", "not_spam"])
        data["text"].append(text)
        data["label"].append(label)
    
    return data

data = generate_data()
X = data["text"]
y = data["label"]


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy}")

# Пример текста на английском для проверки
test_text = "Congratulations! You have won $1000000 in our lottery. To claim your prize, please send your personal information and bank card number to prize@scam.com."
test_text2 = "She is the best girls and she is name Diana"

test_text_vectorized = vectorizer.transform([test_text])
predicted_label = knn.predict(test_text_vectorized)

# Определение результата
if predicted_label[0] == "spam":
    print("Текст классифицируется как спам.")
else:
    print("Текст классифицируется как не спам.")

test_text_vectorized = vectorizer.transform([test_text2])
predicted_label = knn.predict(test_text_vectorized)

if predicted_label[0] == "spam":
    print("Текст классифицируется как спам.")
else:
    print("Текст классифицируется как не спам.")