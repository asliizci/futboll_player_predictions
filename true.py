import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Veri yükleme
file_path = r"C:\Users\gozde\OneDrive\Masaüstü\kod\düzelt\remaining_data.csv"  # Dosya yolunuzu burada belirtin
data = pd.read_csv(file_path)

# Convert 'Value' column to millions
data['Value_M'] = data['Value'] / 1_000_000

# Tüm özellikleri ve hedef değişkeni belirleme
all_features = [
    'Age', 'Overall', 'Potential', 'Dribbling', 'Finishing', 'ShortPassing',
    'LongPassing', 'Composure', 'Vision', 'Acceleration', 'SprintSpeed',
    'International Reputation', 'BMI'
]
target = 'Value_M'

# Veriyi temizleme
data_cleaned = data[all_features + [target]].dropna()

# Hedef değişkeni logaritmik dönüşüm
data_cleaned['Log_Value_M'] = np.log1p(data_cleaned[target])

# Yeni hedef değişkeni belirleme
target_transformed = 'Log_Value_M'

# Özellikler ve hedef değişkeni ayırma
X = data_cleaned[all_features]
y = data_cleaned[target_transformed]

# Özellikleri normalize etme
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Veriyi eğitim, doğrulama ve test setlerine bölme
X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Random Forest modeli eğitme
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Test ve validation setlerinde tahmin yapma
y_test_pred = rf_model.predict(X_test)
y_val_pred = rf_model.predict(X_val)

# Performansı değerlendirme
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"Test MSE: {mse_test:.4f}")
print(f"Test R²: {r2_test:.4f}")
print(f"Validation MSE: {mse_val:.4f}")
print(f"Validation R²: {r2_val:.4f}")

# Yeni oyuncunun tahmin değerini hesaplama
print("Enter the details of the new player:")
new_player = []
for feature in all_features:
    value = float(input(f"Enter {feature}: "))
    new_player.append(value)

# Yeni oyuncunun özelliklerini normalize etme
new_player_df = pd.DataFrame([new_player], columns=all_features)  # Veri çerçevesi oluşturma
new_player_normalized = scaler.transform(new_player_df)

# Yeni oyuncunun değerini tahmin etme
predicted_value_log = rf_model.predict(new_player_normalized)[0]
predicted_value_original = np.expm1(predicted_value_log)  # Logaritmadan geri dönüşüm

print(f"The predicted value for the player is: {predicted_value_original:.2f}M")
