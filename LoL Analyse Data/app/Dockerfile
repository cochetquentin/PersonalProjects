# Utiliser une image de base officielle Python
FROM python:3.10.13-alpine3.19

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances et installer les dépendances
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY static ./static
COPY templates ./templates
COPY tools.py ./
COPY app.py ./

# Exposer le port sur lequel l'application s'exécute
EXPOSE 5000

# Définir la commande pour démarrer l'application
CMD ["python", "app.py"]
