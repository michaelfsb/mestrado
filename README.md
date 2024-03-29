# Optimal control of green hydrogen production

This repository contains the developed codes for my master's thesis. 

The main files are ocp-X-YYY.py, where X is the identification problem and YYY is the method of transcription used.

You can find the results, the Ipopt log, and a graph of control variables and states in the results' folder.

## Problems definition
### Optimal control problem 1
- ocp-1-YYY.py

The system is composed of a photovoltaic panel and a PEM electrolyzer. The control variable $i_{el}$ is the electrolyzer current. 

$\text{Cost} = \int_{0}^{T_f}(N_{el} \cdot v_{el}(t) \cdot i_{el}(t) - v_{pv}(t) \cdot i_{pv}(t))^2\text{d}t$


### Optimal control problem 2
- ocp-2-YYY.py

The system is composed of a photovoltaic panel and a PEM electrolyzer. The control variables are the current $i_{el}$ and the state of the electrolyzer $s_{el}$ (on or idle).

$\text{Cost} = \int_{0}^{T_f}s_{el}(t)\cdot(I_{min}-i_{el}(t))\cdot(N_{el} \cdot v_{el}(t) \cdot i_{el}(t) - v_{pv}(t) \cdot i_{pv}(t))^2\text{d}t$

$f_{H_{2}}(t) = s_{el}(t) \cdot N_{el} \cdot \frac{i_{el}(t)}{F}$

### Optimal control problem 3
- ocp-3-YYY.py

The system is composed of a photovoltaic panel and a PEM electrolyzer. The control variable $i_{el}$ is the electrolyzer current.

$\text{Cost} = R \cdot v_{H_{2}}(T_f) + \int_{0}^{T_f}(N_{el} \cdot v_{el}(t) \cdot i_{el}(t) - v_{pv}(t) \cdot i_{pv}(t))^2\text{d}t$

$f_{H_{2}}(t) = 0 \quad \text{if}~~i_{el}(t) < I_{min}$

$f_{H_{2}}(t) = N_{el} \cdot \frac{i_{el}(t)}{F} \quad \text{if}~~i_{el}(t) \ge I_{min}$

## Dependencies

- casadi
- matplotlib.pyplot
- numpy
- scipy
- scipy.io

## How to rum

To create the environment and resolve the necessary dependencies, rum these commands:

```
conda env create -f environment.yml
conda activate mestrado
```

Then you can run each file or the flow command to run all of them.

```
ls *.py|xargs -n 1 -P 1 python3
```
