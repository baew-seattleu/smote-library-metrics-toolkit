
# Equations Used in the Metric Calculations

Let MI be real minority samples, MA be real majority samples, and SM be synthetic minority samples.

1) Mean difference (%)
MeanDiff% = average over features of:
|mu_SM - mu_MI| / max(|mu_MI|, eps) * 100

2) Std difference (%)
StdDiff% = average over features of:
|sigma_SM - sigma_MI| / max(|sigma_MI|, eps) * 100

3) KL divergence
For each feature i:
KL(P_i || Q_i) = sum_b P_i(b) log(P_i(b)/Q_i(b))
Average over all features.

4) KDE area difference
For each feature i:
Delta_KDE(i) = integral |f_MI,i(x) - f_SM,i(x)| dx
Average over all features.

5) Euclidean distance
ED(p,q) = sqrt(sum_i (p_i - q_i)^2)

6) Hassanat distance
If min(p_i,q_i) >= 0:
D = 1 - (1 + min)/(1 + max)
Else:
D = 1 - (1 + min + |min|)/(1 + max + |min|)

HD(p,q) = sum_i D(p_i,q_i)

7) Geometric Invasion Rate (GIR)
For each synthetic sample s:
d_MI(s) = min over x in MI, x != s of d(s,x)
d_MA(s) = min over y in MA, y != s of d(s,y)

Invalid if:
d_MA(s) < d_MI(s)

GIR = number of invalid synthetic samples / number of synthetic samples

Exact duplicate matches are excluded by replacing zero distances with +infinity before taking the minimum.
