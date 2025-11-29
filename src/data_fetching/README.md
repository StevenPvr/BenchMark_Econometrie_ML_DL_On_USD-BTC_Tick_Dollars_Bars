# Data Fetching Module

Module de telechargement des donnees tick BTC/USD depuis les exchanges crypto via ccxt.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `fetching.py` | Logique principale de telechargement avec rate limiting et parallelisation |
| `main.py` | Point d'entree avec mode daemon et auto-increment des dates |
| `__init__.py` | Export de la fonction principale |

## Fonctionnalites

### Rate Limiting
- Limite globale de 6000 requetes/minute (limite Binance)
- Thread-safe avec partage entre tous les workers
- Attente automatique si limite atteinte

### Telechargement Parallele
- 6 workers par defaut pour maximiser le debit
- Chunks de 2 semaines pour robustesse
- Reprise automatique en cas d'erreur

### Stockage
- Format Parquet avec compression Snappy
- Dataset partitionne (append sans recharger les donnees existantes)
- Deduplication automatique des trades

## Utilisation

### Execution unique
```bash
python -m src.data_fetching.main
```

### Mode daemon (boucle infinie)
```bash
python -m src.data_fetching.main --daemon --delay 300
```

### Options
- `--daemon` : Mode boucle infinie avec recovery d'erreurs
- `--max-iterations N` : Limite le nombre d'iterations
- `--delay N` : Delai entre iterations (secondes, defaut: 300)
- `--no-increment` : Desactive l'auto-increment des dates

## Configuration

Les parametres sont definis dans `src/constants.py` :
- `EXCHANGE_ID` : Exchange source (defaut: binance)
- `SYMBOL` : Paire de trading (defaut: BTC/USDT)
- `START_DATE` / `END_DATE` : Plage de dates

## Sortie

Les donnees sont sauvegardees dans le chemin defini par `DATASET_RAW_PARQUET` dans `src/path.py`.

Colonnes principales :
- `timestamp` : Horodatage UTC du trade
- `price` : Prix d'execution (float32)
- `amount` : Volume (float32)
- `side` : buy/sell (category)
