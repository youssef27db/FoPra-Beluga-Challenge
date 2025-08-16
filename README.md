# FoPra – Beluga Challenge  
**Hierarchischer Reinforcement-Learning-Agent**

---

## � Inhaltsverzeichnis

### 🚀 Erste Schritte
- [Setup & Installation](setup/README.md) - Einrichtung der Entwicklungsumgebung und Installation aller Abhängigkeiten

### 📖 Dokumentation
- [HTML-Dokumentation](https://youssef27db.github.io/FoPra-Beluga-Challenge) - Vollständige generierte Code-Dokumentation

### 🤖 Reinforcement Learning
- [RL-Agent README](rl/README.md) - Detaillierte Anleitung zum Training und zur Verwendung des RL-Agenten
- [MCTS Implementation](rl/mcts/README.md) - C++ und Python Implementierung des Monte Carlo Tree Search

---

## �📌 Projektbeschreibung
Dieses Projekt implementiert einen **hierarchischen RL-Agenten** für die *Beluga Challenge*.  
Der Ansatz trennt strategische Entscheidungen (High-Level) von der konkreten Ausführung (Low-Level), um unterschiedliche Methoden wie **PPO**, **Heuristiken** und **MCTS** gezielt zu kombinieren.

---

## 🏗 Architekturübersicht

Unsere Architektur folgt einem **hierarchischen Entscheidungsansatz**:

![Agent Architecture](docs/architektur.jpg)

- **High-Level Agent**  
  - Wählt eine von acht möglichen Aktionen  
  - Trainiert mit **Proximal Policy Optimization (PPO)**  
  - Verantwortlich für die strategische Richtung

- **Low-Level Agent**  
  - Verfeinert und führt die gewählte High-Level-Aktion aus  
  - Setzt je nach Komplexität unterschiedliche Mechanismen ein:  
    - 🔹 **Direkte Ausführung** – deterministische Aktionen ohne Parameter  
    - 🔹 **Heuristiken** – einfache parametrisierte Aktionen  
    - 🔹 **Monte Carlo Tree Search (MCTS)** – komplexe, sequenzielle Entscheidungen mit großem Kombinationsraum

### 🎯 Die 8 verfügbaren Aktionen:
1. **`load_beluga`** - Beluga mit Jigs vom Trailer beladen
2. **`unload_beluga`** - Jigs aus dem Beluga entladen  
3. **`get_from_hangar`** - Jigs aus dem Hangar holen
4. **`deliver_to_hangar`** - Jigs zum Hangar transportieren
5. **`left_stack_rack`** - Jigs auf dem linken Rack stapeln
6. **`right_stack_rack`** - Jigs auf dem rechten Rack stapeln
7. **`left_unstack_rack`** - Jigs vom linken Rack entstapeln
8. **`right_unstack_rack`** - Jigs vom rechten Rack entstapeln

> 💡 **Vorteil:** Durch diese modulare Trennung können wir die Stärken verschiedener Methoden gezielt nutzen und die Charakteristika einzelner Teilprobleme optimal abdecken.

---

## 📄 Lizenz

Dieses Projekt ist unter der **MIT-Lizenz** lizenziert.

```
MIT License

Copyright (c) 2025 Youssef Daoudi, Jan Kirschbaum, Nils Schulze

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


