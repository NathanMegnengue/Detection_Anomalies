# detection_anomalies.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Charger les données
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Visualiser les données
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Passengers'], label='Passagers')
plt.title('Données de Passagers Aériens')
plt.xlabel('Date')
plt.ylabel('Nombre de Passagers')
plt.legend()
plt.show()

# Détection d'anomalies
model = IsolationForest(contamination=0.05)  # 5% d'anomalies
data['anomaly'] = model.fit_predict(data[['Passengers']])

# Visualiser les anomalies
anomalies = data[data['anomaly'] == -1]
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Passengers'], label='Passagers')
plt.scatter(anomalies.index, anomalies['Passengers'], color='red', label='Anomalies')
plt.title('Détection d\'Anomalies dans les Données de Passagers Aériens')
plt.xlabel('Date')
plt.ylabel('Nombre de Passagers')
plt.legend()
plt.show()
