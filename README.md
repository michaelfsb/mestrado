# Optimal control of green hydrogen production

This repository contains the developed codes for my master's thesis. 

The main files are ocp-X-YYY.py, where X is the identification problem and YYY is the method of transcription used.

You can find the results, the Ipopt log, and a graph of control variables and states in the results' folder.

## Optimal control problem 1
- ocp-1-YYY.py

The system is composed of a photovoltaic panel and a PEM electrolyzer. The control variable $i_{el}$ is the electrolyzer current. 

$\text{Cost} = \int_{0}^{T_f}(N_{el} \cdot v_{el}(t) \cdot i_{el}(t) - v_{pv}(t) \cdot i_{pv}(t))^2\text{d}t$


## Optimal control problem 2
- ocp-2-YYY.py

The system is composed of a photovoltaic panel and a PEM electrolyzer. The control variables are the current $i_{el}$ and the state of the electrolyzer $p_{el}$ (on or idle).
