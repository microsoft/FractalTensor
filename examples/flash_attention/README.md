# Algorithm Idea

## The intuition

Given a query $q_i \in \mathbb{R}^d$, and a lists of keys and values $k_1,\cdots,k_n$ and $v_1, \cdots, v_n \in \mathbb{R}^d$ of length $n$


$$\begin{align}
s_i &= \text{dot}(q, k_i) \\
s_{i}' &= \frac{e^{s_i}}{\sum_je^{s_j}} \\
\text{attention}(q,k,v) &= \sum_i{v_is_i'} \\
\end{align}$$

The summation in equation (2) could be moved to the very end of the attention operation (3)

$$\begin{align*}
s_i &= \text{dot}(q, k_i) \\
s_{i}' &= e^{s_i} \\
\text{attention}(q,k,v) &= \frac{\sum_i{v_is_i'}}{\sum_je^{s_j}}
\end{align*}$$

The processing process can be written as:

$$\begin{align*}
s_i &= \text{dot}(q,k_i)  \\
v^* &\leftarrow v^* + v_ie^{s_i} \\
s^* &\leftarrow s^* + e^{s_i} \\
\text{attention}(q,k,v) &= \frac{v^*}{s^*} 
\end{align*}$$

## Numerical Stability

intialize $v^* \in \mathbb{R}^d = 0$, $s* \in R = 0$, $m = -\text{inf}$

$$\begin{align*}
s_i &= \text{dot}(q,k_i)  \\
m_i &= \text{max}(m^*,s_i)\\
v^* &\leftarrow v^*e^{m^*-m_i} + v_ie^{s_i-m_i} \\
s^* &\leftarrow s^*e^{m^*-m_i} + e^{s_i-m_i} \\
m^* &\leftarrow m_i \\
\text{attention}(q,k,v) &= \frac{v^*}{s^*} \\
\end{align*}$$

# Reference

1. Rabe, Markus N., and Charles Staats. "[Self-attention Does Not Need $ O (n^ 2) $ Memory](https://arxiv.org/pdf/2112.05682.pdf)." arXiv preprint arXiv:2112.05682 (2021).
