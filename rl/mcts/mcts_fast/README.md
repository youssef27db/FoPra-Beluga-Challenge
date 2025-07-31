# MCTS_Fast Python Module

Ein schnelles Monte Carlo Tree Search (MCTS) Modul mit C++ Backend und Python-Bindings via pybind11.

## Einfache Installation

Die einfachste Installation erfolgt über die mitgelieferten Installationsskripte:

### macOS/Linux:
```bash
# Im Hauptverzeichnis des Projekts
./install_mcts_fast.sh
```

### Windows:
```bash
# Im Hauptverzeichnis des Projekts - In einer Developer Command Prompt ausführen
install_mcts_fast.bat
```

Diese Skripte installieren automatisch alle notwendigen Abhängigkeiten inklusive OpenMP (falls nötig).

## Manuelle Installation

### Methode 1: Installation direkt aus dem Repository

```bash
# Von der Hauptverzeichnisebene des Projekts
pip install ./rl/mcts/mcts_fast
```

### Methode 2: Installation von vorgebauten Wheels (wenn verfügbar)

```bash
# Wird verfügbar sein, wenn GitHub Actions eingerichtet ist
pip install mcts_fast
```

#### Was sind Wheels und wozu sind sie nützlich?

Wheels sind vorkompilierte Pakete - wie "Installationsprogramme" für Python-Module mit folgenden Vorteilen:

1. **Keine Kompilierung nötig**: Du brauchst keinen eigenen C++-Compiler
2. **Schnelle Installation**: Das Paket ist bereits gebaut und muss nur installiert werden
3. **Plattform-spezifisch**: Es gibt unterschiedliche Wheels für Windows, macOS und Linux

Wheels dienen als:
- **Backup-Option**: Wenn die direkte Installation nicht funktioniert
- **Einfache Distribution**: Du kannst das passende Wheel direkt installieren
- **Kompatibilitätssicherung**: Vermeidet lokale Kompilierungsprobleme

#### So nutzt du Wheels (falls nötig):

Wenn du Probleme mit der regulären Installation hast:

1. Lade das passende Wheel für dein System herunter (falls verfügbar)
2. Installiere es mit:
   ```bash
   pip install mcts_fast-0.1.0-cp39-cp39-win_amd64.whl
   ```
   (Der genaue Name hängt von Python-Version und Plattform ab)

## Abhängigkeiten

### Python-Abhängigkeiten
Diese werden automatisch installiert:
- Python 3.6+
- NumPy
- pybind11

### C++-Abhängigkeiten

#### OpenMP
- **Windows**: Kommt automatisch mit Visual Studio C++ Build Tools
- **macOS**: Muss manuell installiert werden: `brew install libomp`
- **Linux**: Auf Ubuntu/Debian: `sudo apt-get install libomp-dev`

#### Compiler
Ein C++-Compiler mit C++17-Unterstützung:
- **macOS**: clang (Xcode Command Line Tools)
- **Windows**: Visual Studio 2019+ mit C++ Build Tools
- **Linux**: GCC 7+ oder Clang 5+

## Betriebssystem-spezifische Hinweise

### macOS
1. Xcode Command Line Tools installieren: `xcode-select --install`
2. OpenMP installieren: `brew install libomp`
3. Das Installationsskript ausführen: `./install_mcts_fast.sh`

### Windows

#### Schritt-für-Schritt Anleitung für Windows-Nutzer:

1. **Lade das Repository herunter** oder mache ein `git clone`
2. **Installiere Visual Studio mit C++ Build Tools**:
   - Lade Visual Studio Community Edition von [hier](https://visualstudio.microsoft.com/de/vs/community/) herunter
   - Wähle bei der Installation "Desktop development with C++" aus
3. **Öffne eine Developer Command Prompt**:
   - Drücke Start, suche nach "Developer Command Prompt for VS" und öffne es
4. **Navigiere zum Projekt-Ordner**:
   ```bash
   cd Pfad\zu\FoPra-Beluga-Challenge
   ```
5. **Führe das Installationsskript aus**:
   ```bash
   install_mcts_fast.bat
   ```
6. **Teste die Installation**:
   ```bash
   python -c "import mcts_fast; print('Import erfolgreich')"
   ```

### Linux
1. Kompiler installieren: `sudo apt-get install build-essential`
2. OpenMP installieren: `sudo apt-get install libomp-dev`
3. Das Installationsskript ausführen: `./install_mcts_fast.sh`

## Entwicklung

Für Entwickler ist es empfehlenswert, das Modul im Development-Modus zu installieren:

```bash
pip install -e ./rl/mcts/mcts_fast
```

## Fehlerbehebung

### Allgemeine Probleme
1. Aktualisieren Sie pip und setuptools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

### macOS-spezifische Probleme
1. Falls Fehler mit OpenMP auftreten: `brew reinstall libomp`
2. Bei Compiler-Problemen: `xcode-select --install` erneut ausführen

### Windows-spezifische Probleme
1. Stellen Sie sicher, dass Sie die Command Prompt als Administrator ausführen
2. Versuchen Sie die Installation ohne Build-Isolation:
   ```bash
   pip install --no-build-isolation ./rl/mcts/mcts_fast
   ```
3. Bei Fehlern "Microsoft Visual C++ 14.0 or greater is required":
   - Installieren Sie Visual Studio mit C++ Build Tools neu
   - Stellen Sie sicher, dass Sie eine Developer Command Prompt verwenden
   - Alternativ können Sie das [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) Paket separat installieren

### Linux-spezifische Probleme
1. Falls Fehler mit OpenMP: `sudo apt-get install --reinstall libomp-dev`
2. Bei Compiler-Fehlern: `sudo apt-get install --reinstall build-essential`
