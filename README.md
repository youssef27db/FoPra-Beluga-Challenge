# Architekturübersicht: Hierarchischer RL-Agent für die Beluga Challenge

Unsere Architektur basiert auf einem hierarchischen Entscheidungsansatz. Der High-Level-Agent trifft strategische Entscheidungen, während die Parametrisierung und Ausführung über spezialisierte Low-Level-Mechanismen erfolgt:
![agent-architecture](https://github.com/user-attachments/assets/ac2d5b83-8f99-4fbd-b97b-1441114ee30b)



## Komponenten

- **High-Level Agent**  
  Wählt eine von acht möglichen Aktionen aus (z. B. *Load Jig*, *Swap*, *Dispatch*). Trainiert mit Proximal Policy Optimization (PPO).

- **Low-Level Agent**  
  Verfeinert und führt die vom High-Level-Agenten gewählte Aktion aus. Abhängig von der Aktionsart geschieht dies durch:
  - 🔹 *Direkte Ausführung* (z. B. deterministisch lösbare Aktionen ohne Parameter)
  - 🔹 *Heuristiken* (für einfache, aber parametrisierte Aktionen)
  - 🔹 *Monte Carlo Tree Search (MCTS)* (für komplexe, sequenzielle Entscheidungen mit hohem Kombinationsraum)

Diese modulare Trennung erlaubt es, unterschiedliche Ansätze (RL, Heuristiken, MCTS) synergetisch zu kombinieren und gezielt auf die Charakteristika einzelner Teilprobleme anzuwenden.
