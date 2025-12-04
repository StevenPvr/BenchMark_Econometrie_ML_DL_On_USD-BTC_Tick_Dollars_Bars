# Commandes Terminal Utiles

## Lancer un programme en arrière-plan

```bash
# Avec &
python mon_script.py &

# Avec nohup (continue même si terminal fermé)
nohup python mon_script.py > output.log 2>&1 &
```

## tmux - Sessions persistantes

```bash
# Installer
brew install tmux

# Nouvelle session nommée
tmux new -s nom_session

# Détacher (sans arrêter) : Ctrl+B, puis D

# Lister les sessions
tmux ls

# Revenir à une session
tmux attach -t nom_session

# Tuer une session
tmux kill-session -t nom_session
```

## Raccourcis Terminal macOS

- `Cmd + T` : Nouvel onglet
- `Cmd + N` : Nouvelle fenêtre
