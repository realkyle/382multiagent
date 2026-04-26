# Multi-Agent Pacman

Implementation of multi-agent search algorithms for the classic Pacman game (UC Berkeley CS188 Project 2).

## Overview

All agent implementations live in `multiAgents.py`.

### Q1 — Reflex Agent (4/4 pts)
Improved evaluation function that scores state-action pairs using:
- Reciprocal distance to nearest food (`10 / dist`) to incentivize eating
- Heavy penalty for being adjacent to active ghosts
- Bonus for proximity to scared ghosts (chase them for points)

Achieves 10/10 wins on `openClassic` with an average score of ~1225.

### Q2 — Minimax Agent (5/5 pts)
Full minimax search to arbitrary depth with any number of ghosts. One "ply" = one Pacman move + all ghost moves. Leaves are scored with `self.evaluationFunction`.

### Q3 — Alpha-Beta Pruning (5/5 pts)
Alpha-beta pruning over the same minimax tree. Processes successors in `getLegalActions` order with strict inequality pruning (`> beta` / `< alpha`) to match the autograder exactly.

## Running

```bash
# Play manually
python pacman.py

# Run a specific agent
python pacman.py -p ReflexAgent -l testClassic
python pacman.py -p MinimaxAgent -a depth=3 -l minimaxClassic
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic

# Autograder
python autograder.py
python autograder.py -q q1 --no-graphics
```
