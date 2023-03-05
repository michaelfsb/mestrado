```math
\begin{aligned}
& \underset{u(\cdot)}{\text{minimize}}
& &  \int_{0}^{10} (x_1^2 + x_2^2 + u^2) \,dx  \\
& \text{subject to}
& & \dot x_1 = (1 - x_2^2)x_1 - x_2 + 1, \\
&&& \dot x_2 = x_1, \\
&&& x_1 \geq -0.25, \\
&&& -1 \leq u \leq 1.
\end{aligned}
```
