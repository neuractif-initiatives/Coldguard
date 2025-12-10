import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import plotly.express as px


# Charger le dataset
df = pd.read_csv(r'C:\Users\hp\Downloads\dataset_chaine_froid.csv')

# Transformer date_arrivee en âge du conteneur (en heures)
df['date_arrivee'] = pd.to_datetime(df['date_arrivee'])
df['age_conteneur_heures'] = (datetime.now() - df['date_arrivee']).dt.total_seconds() / 3600
df.drop('date_arrivee', axis=1, inplace=True)

# Encodage des variables catégorielles
label_encoders = {}
for col in ['type_marchandise', 'statut_processus', 'risque_deterioration']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Séparation features / cible
X = df.drop('risque_deterioration', axis=1)
y = df['risque_deterioration']

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Définir les modèles
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Comparaison des performances
results = []
confusion_matrices = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({'Modèle': name, 'Accuracy': acc, 'F1-score': f1})
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

# Affichage des résultats
print(pd.DataFrame(results))
print('Matrice de confusion pour chaque modèle:')
for name, cm in confusion_matrices.items():
        print(f"\n{name}:\n{cm}")


df_results = pd.DataFrame(results)
fig = px.bar(df_results, x='Modèle', y=['Accuracy','F1-score'], barmode='group', title='Comparaison des modèles')
fig.show()



# Prédictions sur le jeu de test
y_pred = models['GradientBoosting'].predict(X_test)

# Décodage des classes pour la visualisation
classes_decoded = label_encoders['risque_deterioration'].inverse_transform(y_pred)



