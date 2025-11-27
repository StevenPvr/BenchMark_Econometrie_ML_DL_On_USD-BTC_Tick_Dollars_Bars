# Data Fetching Daemon - Récupération Automatique de Données

Le daemon de récupération de données permet de collecter automatiquement les données BTC/USD en continu, sans intervention manuelle.

## 🚀 Démarrage Rapide

### Lancer le daemon en arrière-plan :
```bash
./scripts/run_data_fetching_daemon.sh start
```

### Vérifier le statut :
```bash
./scripts/run_data_fetching_daemon.sh status
```

### Consulter les logs en temps réel :
```bash
./scripts/run_data_fetching_daemon.sh logs
```

### Arrêter le daemon :
```bash
./scripts/run_data_fetching_daemon.sh stop
```
# 2. Pour reprendre avec le fichier consolidé comme base
```bash
./scripts/run_data_fetching_daemon.sh base

## 📋 Commandes Disponibles

| Commande | Description |
|----------|-------------|
| `./scripts/run_data_fetching_daemon.sh start` | Démarre le daemon en arrière-plan |
| `./scripts/run_data_fetching_daemon.sh stop` | Arrête le daemon proprement |
| `./scripts/run_data_fetching_daemon.sh restart` | Redémarre le daemon |
| `./scripts/run_data_fetching_daemon.sh status` | Affiche le statut et les derniers logs |
| `./scripts/run_data_fetching_daemon.sh logs` | Suit les logs en temps réel |

## ⚙️ Fonctionnement

### Cycle Automatique :
1. **Récupération** : Télécharge 5 jours de données BTC/USD
2. **Incrémentation** : Avance automatiquement les dates (+5 jours)
3. **Pause** : Attend 5 minutes avant la prochaine itération
4. **Récupération d'erreurs** : Continue même en cas d'erreur

### Gestion des Dates :
- **Début** : Utilise les dates de `constants.py` au premier lancement
- **Incrémentation** : +5 jours automatiquement après chaque succès
- **Persistance** : Sauvegarde l'état dans `data/fetch_dates_state.json`
- **Gestion des mois** : Gère automatiquement les fins de mois (28/30/31 jours)

### Gestion des Erreurs :
- **Récupération automatique** : Continue après les erreurs
- **Backoff exponentiel** : Attend plus longtemps après les erreurs répétées
- **Arrêt de sécurité** : S'arrête après 5 erreurs consécutives
- **Logs détaillés** : Toutes les erreurs sont loggées

## 📁 Fichiers Créés

```
data/
├── fetch_dates_state.json          # État des dates actuelles
├── raw/
│   └── dataset_raw.parquet/        # Données partitionnées
│       ├── part-00000.parquet      # Tes données nettoyées
│       ├── part-*.parquet          # Nouvelles données ajoutées
│       └── ...
└── fetching_daemon.pid             # PID du daemon

logs/
└── data_fetching_daemon.log        # Logs du daemon
```

## 🔧 Configuration

### Modifier la fréquence :
Éditer `src/data_fetching/main.py` :
```python
AUTO_INCREMENT_DAYS: int = 5  # Jours par fenêtre
# Et dans main_loop():
delay_seconds: int = 300       # 5 minutes entre itérations
```

### Modifier le nombre de workers :
```python
max_workers: int = 6  # Workers parallèles (défaut: 6)
```

## 📊 Monitoring

### Logs en temps réel :
```bash
./scripts/run_data_fetching_daemon.sh logs
```

### Vérifier les données collectées :
```bash
# Nombre de fichiers partitionnés
ls -la data/raw/dataset_raw.parquet/ | wc -l

# Taille totale
du -sh data/raw/dataset_raw.parquet/

# Dernières données
python3 -c "
import pandas as pd
df = pd.read_parquet('data/raw/dataset_raw.parquet')
print(f'Données: {len(df):,} trades')
print(f'Période: {df.timestamp.min()} → {df.timestamp.max()}')
"
```

## 🧠 Pipeline Optimisé Mémoire

Le système utilise une approche **zéro consolidation** pour économiser drastiquement la RAM :

### Architecture Intelligente :
1. **Accumulation** : Fichiers parquet partitionnés restent séparés (pas de fusion)
2. **Traitement individuel** : `data_preparation` traite chaque fichier un par un
3. **Réduction drastique** : Convertit **millions de trades → milliers de dollar bars**
4. **Pas de fusion massive** : Évite de charger 72M lignes en mémoire simultanément

### Avantages Mémoire :
- ✅ **Zéro consolidation** des trades bruts (évite 72M lignes en RAM)
- ✅ **Traitement séquentiel** des fichiers (max 50M lignes à la fois)
- ✅ **Compression finale** : Dollar bars = ~1/1000ème de la taille originale
- ✅ **Scalabilité** : Marche avec des datasets de plusieurs milliards de trades

### Comparaison :
```
❌ Ancienne approche : Charger 72M trades → 16GB RAM → Consolidation
✅ Nouvelle approche : 45M trades → 6M trades → 3M trades → 3M trades → Dollar bars
```

## 🛑 Arrêt d'Urgence

Si le daemon ne répond plus :
```bash
# Trouver le PID
ps aux | grep "data_fetching"

# Tuer manuellement
kill -9 <PID>
rm -f data/fetching_daemon.pid
```

## 🎯 Usage Typique

```bash
# Lancer pour collecter des données en continu
./scripts/run_data_fetching_daemon.sh start

# Vérifier régulièrement
./scripts/run_data_fetching_daemon.sh status

# Quand tu as assez de données
./scripts/run_data_fetching_daemon.sh stop
```

Le daemon tournera indéfiniment et collectera automatiquement de plus en plus de données historiques ! 🚀
