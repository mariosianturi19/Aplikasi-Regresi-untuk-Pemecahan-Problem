import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# KODE SUMBER
# Pembacaan data dari file .csv yang di download dari Kaggle=
file_path = "D:\Kuliah\Bahan Kuliah\Matkul\Vscode\Aplikasi Regresi untuk Pemecahan Problem\Student_Performance.csv"
df = pd.read_csv(file_path)

print(df.head())
# Durasi waktu belajar (TB) terhadap nilai ujian (NT)- Problem yang pertama
X = df['Hours Studied'].values.reshape(-1, 1)
y = df['Performance Index'].values

# Model Linear (Metode 1)
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

# Model Pangkat Sederhana (Metode 2)
def power_law(x, a, b):
    return a * np.power(x, b)

params, _ = curve_fit(power_law, df['Hours Studied'], df['Performance Index'])
y_pred_power = power_law(df['Hours Studied'], *params)
rms_power = np.sqrt(mean_squared_error(y, y_pred_power))

plt.figure(figsize=(12, 6))

plt.scatter(df['Hours Studied'], df['Performance Index'], color='blue', label='Data Asli')

plt.plot(df['Hours Studied'], y_pred_linear, color='red', label=f'Model Linear (RMS={rms_linear:.2f})')

plt.plot(df['Hours Studied'], y_pred_power, color='green', label=f'Model Pangkat Sederhana (RMS={rms_power:.2f})')

plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.legend()
plt.title('Regresi Hours Studied terhadap Performance Index')
plt.show()


# KODE TESTING, untuk menampilkan nilai RMS dari setiap model
print(f'RMS Model Linear: {rms_linear:.2f}')
print(f'RMS Model Pangkat Sederhana: {rms_power:.2f}')