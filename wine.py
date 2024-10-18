from sklearn.datasets import load_wine
import pandas as pd

# Wine veri setini yükleme
wine = load_wine()

# Veri setini DataFrame'e dönüştürme
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Hedef (sınıflar) ekleniyor
df['TARGET'] = wine.target

# İlk birkaç satırı görme
print(df.head())


from sklearn.model_selection import train_test_split

# Bağımsız değişkenler (X) ve hedef değişken (y)
X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Veri setini bölme (eğitim ve test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)


from sklearn.linear_model import LogisticRegression

# Modeli oluşturma ve eğitme
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Test veri seti üzerinde tahminler yapma
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score

# Doğruluk oranı hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk oranı: {accuracy}")


# Test verileri üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Tahmin sonuçlarını görüntüleme
print(y_pred)



import matplotlib.pyplot as plt

# Gerçek ve tahmin edilen değerleri karşılaştıran bir grafik oluşturma
plt.scatter(y_test, y_pred)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek ve Tahmin Edilen Değerlerin Karşılaştırılması')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ortalama mutlak hata (MAE) hesaplama
mae = mean_absolute_error(y_test, y_pred)

# Ortalama kare hata (MSE) hesaplama
mse = mean_squared_error(y_test, y_pred)

print(f"Ortalama Mutlak Hata (MAE): {mae}")
print(f"Ortalama Kare Hata (MSE): {mse}")
