import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, jsonify
from flask_cors import CORS
import json # Pour sérialiser les objets non-JSON standard

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






# Créer un DataFrame pour la visualisation
import pandas as pd
import plotly.express as px

df_vis = pd.DataFrame({'Risque prédit': classes_decoded})

# Créer un histogramme de la distribution des risques prédits
fig = px.histogram(df_vis, x='Risque prédit', color='Risque prédit',
                   title='Distribution des prédictions de risque sur le jeu de test',
                   labels={'Risque prédit': 'Risque prédit'},
                   text_auto=True)
fig.show()

#mise en place de l'API

app = Flask(__name__)
CORS(app)  # Autorise les requêtes depuis votre application React


# Fonction pour préparer et retourner TOUS les résultats
def get_ml_dashboard_data(df_results, confusion_matrices, classes_decoded):
    # 1. Préparer les métriques (DataFrame de comparaison)
    metrics_data = df_results.to_dict(orient='records')

    # 2. Préparer les matrices de confusion (doivent être converties en liste/JSON)
    cm_data = {
        name: cm.tolist()  # numpy array doit être converti en liste Python
        for name, cm in confusion_matrices.items()
    }

    # 3. Préparer les données pour l'histogramme de prédiction
    # On compte la fréquence de chaque prédiction
    prediction_counts = pd.Series(classes_decoded).value_counts().reset_index()
    prediction_counts.columns = ['Risque_predit', 'Count']
    prediction_data = prediction_counts.to_dict(orient='records')

    return {
        'metrics': metrics_data,
        'confusionMatrices': cm_data,
        'predictionDistribution': prediction_data,
        # Vous pouvez ajouter ici l'importance des features (Feature Importance) si vous utilisez RF ou GB
    }


# Définir l'endpoint de l'API
@app.route('/api/ml-results', methods=['GET'])
def serve_ml_results():
    # Ici, nous nous assurons que les variables sont disponibles après la logique ML
    # Assurez-vous que df_results, confusion_matrices, classes_decoded sont calculées avant

    # Appel de votre logique ML...
    # ... (Si ce code était dans une fonction, vous l'appelleriez ici)

    # Puis, appelez la fonction pour formater les données
    final_data = get_ml_dashboard_data(df_results, confusion_matrices, classes_decoded)

    return jsonify(final_data)


if __name__ == '__main__':
    # Lancez le serveur Flask après l'exécution de tout votre code ML
    print("Démarrage du serveur Flask sur http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)


