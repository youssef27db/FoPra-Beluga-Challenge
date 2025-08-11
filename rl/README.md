# Reinforcement Learning für Beluga Challenge

Dieses Verzeichnis enthält die Reinforcement Learning Lösung für die Beluga Challenge - ein Containerschiff-Optimierungsproblem mit PPO (Proximal Policy Optimization) Agent und MCTS.

## Überblick

Die Lösung verwendet einen hierarchischen Ansatz mit:
- **High-Level Agent**: PPO Agent für strategische Entscheidungen
- **Low-Level Heuristiken**: Für spezifische Containerplatzierung
- **MCTS**: Monte Carlo Tree Search für Exploration
- **Environment**: Simulation der Beluga Challenge Umgebung

## Verzeichnisstruktur

```
rl/
├── agents/           # Agent-Implementierungen
│   ├── high_level/   # PPO Agent
│   └── low_level/    # Heuristische Agents
├── env/              # Umgebung und Zustandsrepräsentation
├── mcts/             # Monte Carlo Tree Search
├── training/         # Training-Logic
├── utils/            # Hilfsfunktionen
└── main.py           # Haupteinstiegspunkt
```

## Verwendung

### Hilfe anzeigen
```bash
python -m rl.main --help
```

### Training

#### Basis Training
```bash
python -m rl.main --mode train
```

#### Training mit benutzerdefinierten Parametern
```bash
python -m rl.main --mode train --n_episodes 5000 --base_index 61
```

#### Training mit Permutation
```bash
python -m rl.main --mode train --n_episodes 5000 --base_index 61 --use_permutation
```
### Evaluierung

#### Modell Evaluierung
```bash
python -m rl.main --mode eval
```

#### Evaluierung mit Plot
```bash
python -m rl.main --mode eval --n_eval_episodes 20 --plot
```

#### Evaluierung mit benutzerdefinierten Parametern
```bash
python -m rl.main --mode eval --n_eval_episodes 50 --max_steps 300
```

### Problem-spezifische Evaluierung

#### Evaluierung eines spezifischen Problems
```bash
python -m rl.main --mode problem --problem_path "problems/problem_7_s49_j5_r2_oc85_f6.json"
```

#### Evaluierung mit Ergebnis-Speicherung
```bash
python -m rl.main --mode problem --problem_path "problems/problem_7_s49_j5_r2_oc85_f6.json" --save_to_file
```

#### Evaluierung eines großen Problems
```bash
python -m rl.main --mode problem --problem_path "problems/problem_90_s132_j137_r8_oc81_f43.json" --max_problem_steps 50000 --save_to_file
```

## Parameter Übersicht

### Allgemeine Parameter
- `--mode`: Modus (`train`, `eval`, `problem`)
- `--base_index`: Basis Index für Problemauswahl (Standard: 61)

### Training Parameter
- `--train_old_models`: Bestehende Modelle laden (Standard: True)
- `--use_permutation`: Observation Permutation verwenden (Standard: False)
- `--n_episodes`: Anzahl Trainingsepisoden (Standard: 10000)

### Evaluierungs Parameter
- `--n_eval_episodes`: Anzahl Evaluierungsepisoden (Standard: 10)
- `--max_steps`: Maximale Schritte pro Episode (Standard: 200)
- `--plot`: Plot nach Evaluierung anzeigen (Standard: False)

### Problem Evaluierungs Parameter
- `--problem_path`: Pfad zum Problem (Standard: "problems/problem_90_s132_j137_r8_oc81_f43.json")
- `--max_problem_steps`: Maximale Schritte für Problemevaluierung (Standard: 20000)
- `--save_to_file`: Ergebnisse in TXT-Datei speichern (Standard: False)


## Modell-Konfiguration

Der PPO Agent wird mit folgenden Standardparametern konfiguriert:
- **Aktionen**: 8 mögliche Aktionen
- **Batch Size**: 128
- **Epochen**: 5
- **Lernrate**: 0.0005
- **Buffer Size**: 1024
- **Policy Clip**: 0.2

## Ausgabedateien

Bei Verwendung von `--save_to_file` werden die Ergebnisse in TXT-Dateien im "results" Verzeichnis gespeichert.

## Troubleshooting

### Häufige Probleme
1. **Import Fehler**: Sicherstellen, dass alle Abhängigkeiten installiert sind
2. **Pfad Probleme**: Programm aus rl.main ausführen

## Analyse

![Bild1](../docs/Bild1.svg)

![Bild2](../docs/Bild2.svg)

![Bild3](../docs/Bild3.svg)