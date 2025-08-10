# 🚀 Beluga Challenge - Installation Setup

Dieses Verzeichnis enthält alle notwendigen Dateien zur Installation der Abhängigkeiten für das Beluga Challenge Projekt.

## 📁 Dateien

### 📋 Requirements Dateien
- **`requirements.txt`** - Vollständige Liste aller Abhängigkeiten mit detaillierten Kommentaren
- **`requirements_simple.txt`** - Einfache Liste ohne Kommentare für automatisierte Installation

### 🖥️ Installationsskripte

#### Windows
- **`install_dependencies.bat`** - Windows Batch-Skript für automatische Installation
- **`setup_dependencies.py`** - Plattformunabhängiges Python-Skript mit detailliertem Feedback

#### Linux/Ubuntu
- **`install_dependencies.sh`** - Shell-Skript für Ubuntu/Linux mit Virtual Environment Support

## 🛠️ Installation

### Option 1: Automatische Installation (Windows)
```cmd
# Einfach das Batch-Skript ausführen
install_dependencies.bat
```

### Option 2: Automatische Installation (Ubuntu/Linux)
```bash
# Ausführungsrechte setzen und Skript ausführen
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### Option 3: Python Setup-Skript (Plattformunabhängig)
```bash
python setup_dependencies.py
```

### Option 4: Manuelle Installation
```bash
# Einfache Installation
pip install -r requirements_simple.txt

# Oder mit detaillierter Ausgabe
pip install -r requirements.txt
```

## 📋 Hauptabhängigkeiten

| Bibliothek | Version | Zweck |
|------------|---------|-------|
| PyTorch | ≥2.0.0 | Neuronale Netzwerke (PPO Agent) |
| NumPy | ≥1.21.0 | Numerische Berechnungen |
| Matplotlib | ≥3.5.0 | Visualisierung |
| Gymnasium | ≥0.29.0 | RL Environment API |

## 🐍 Python Virtual Environment (Empfohlen)

### Windows
```cmd
python -m venv beluga_env
beluga_env\Scripts\activate
pip install -r requirements_simple.txt
```

### Linux/Ubuntu
```bash
python3 -m venv beluga_env
source beluga_env/bin/activate
pip install -r requirements_simple.txt
```

## 🚀 Nach der Installation

Nach erfolgreicher Installation können Sie das Projekt verwenden:

```bash
# Hilfe anzeigen
python -m rl.main --help

# Training starten
python -m rl.main --mode train

# Problem evaluieren
python -m rl.main --mode problem --problem_path problems/problem_7_s49_j5_r2_oc85_f6.json

# Problem evaluieren mit Ausgabe in Datei
python -m rl.main --mode problem --problem_path problems/problem_7_s49_j5_r2_oc85_f6.json --save_to_file
```

## 🔧 Fehlerbehebung

### Häufige Probleme

1. **PyTorch Installation schlägt fehl**
   - Überprüfen Sie Ihre Python-Version (≥3.8 erforderlich)
   - Bei GPU-Unterstützung: Besuchen Sie https://pytorch.org/get-started/locally/

2. **Gymnasium Installation Probleme**
   - Versuchen Sie: `pip install --upgrade setuptools wheel`
   - Dann: `pip install gymnasium`

### System-spezifische Anforderungen

#### Ubuntu/Linux
```bash
sudo apt update
sudo apt install python3-dev build-essential libssl-dev libffi-dev
```

#### Windows
- Microsoft Visual C++ 14.0 oder höher (für einige Pakete)
- Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## 📞 Support

Bei Problemen mit der Installation:
1. Überprüfen Sie die Python-Version: `python --version`
2. Aktualisieren Sie pip: `pip install --upgrade pip`
3. Verwenden Sie ein Virtual Environment
4. Konsultieren Sie die offizielle Dokumentation der jeweiligen Bibliothek

---

*Erstellt für das Beluga Challenge Projekt - FortgeschrittenenPraktikum*
