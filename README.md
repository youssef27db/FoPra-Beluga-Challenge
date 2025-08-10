# FoPra – Beluga Challenge  
**Hierarchischer Reinforcement-Learning-Agent**

---

## 📌 Projektbeschreibung
Dieses Projekt implementiert einen **hierarchischen RL-Agenten** für die *Beluga Challenge*.  
Der Ansatz trennt strategische Entscheidungen (High-Level) von der konkreten Ausführung (Low-Level), um unterschiedliche Methoden wie **PPO**, **Heuristiken** und **MCTS** gezielt zu kombinieren.

---

## 🏗 Architekturübersicht

Unsere Architektur folgt einem **hierarchischen Entscheidungsansatz**:

![Agent Architecture](docs/architektur.jpg)

- **High-Level Agent**  
  - Wählt eine von acht möglichen Aktionen (z. B. *Load Jig*, *Swap*, *Dispatch*)  
  - Trainiert mit **Proximal Policy Optimization (PPO)**  
  - Verantwortlich für die strategische Richtung

- **Low-Level Agent**  
  - Verfeinert und führt die gewählte High-Level-Aktion aus  
  - Setzt je nach Komplexität unterschiedliche Mechanismen ein:  
    - 🔹 **Direkte Ausführung** – deterministische Aktionen ohne Parameter  
    - 🔹 **Heuristiken** – einfache parametrisierte Aktionen  
    - 🔹 **Monte Carlo Tree Search (MCTS)** – komplexe, sequenzielle Entscheidungen mit großem Kombinationsraum

> 💡 **Vorteil:** Durch diese modulare Trennung können wir die Stärken verschiedener Methoden gezielt nutzen und die Charakteristika einzelner Teilprobleme optimal abdecken.

## 📄 Projektdokumentation
Die vollständige generierte HTML-Dokumentation findest du hier:  
[📖 Projekt-Dokumentation anzeigen](https://youssef27db.github.io/FoPra-Beluga-Challenge)


