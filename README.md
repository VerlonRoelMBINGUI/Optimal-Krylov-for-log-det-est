# Optimal-stochastic-Krylov-methods-for-large-scale-log-determinant-estimation

This repository contained the code to evalate the log-determinant of  sparse SPD large matrices based on two methods, Optimal stochastic Arnolidi incompte orthogognalization (OSA-IOP) and Optimal stochastic lanczos quadrature method (OSLQ). 

# OSA-IOP logdet

Estimate **logdet(Q)=tr(log(Q))**  of large sparse **SPD** matrices using:

- **Arnoldi Incomplete orthogonalization procedure ** to approximate the matrix function action `log(Q) v`
- **Hutch++** to estimate `tr(log(Q))` using only matrix–vector products
  
# OSLQ logdet

Estimate **logdet(Q)=tr(log(Q))**  of large sparse **SPD** matrices using:

- **SLQ [2] ** to approximate the matrix function action `v^Tlog(Q) v`
- **Hutch++** to estimate `tr(log(Q))`.


This is useful in large-scale Gaussian models / GMRFs where log-determinants appear in likelihoods, numerical linear Algebra and Machine Learning.
- We used the data from University of Florida [1]
---

## Features

- Works with **sparse** matrices (`scipy.sparse`) and supports **matrix-free** `matvec` callables
- Gershgorin-based spectral interval estimate `[λ_min, λ_max]`
- PSD-safe shifting:
- OSA-IOP
- OSLQ


## Cite our paper
<pre> 
 @article{mbingui2026optimal,
  title={Optimal Stochastic Krylov based Techniques for Large-Scale Log-Determinant Estimation},
  author={Mbingui, Verlon Roel and Tambue, Antoine and Karambal, Issa},
  journal={arXiv preprint arXiv:2606.07004},
  year={2026}
}
<pre> 
## References 
[1] Davis, T.A., Hu, Y., 2011. The university of florida sparse matrix collection. ACM Transactions on Mathematical Software (TOMS) 38, 1–25.

[2] S. Ubaru, J. Chen, and Y. Saad, Fast estimation of tr(f(a)) via stochastic lanczos quadrature, SIAM Journal on Matrix Analysis and Applications, 38 (2017), pp. 1075–1099.

[3] Z. Han, W. Li, Y. Huang, and S. Zhu, Suboptimal subspace construction for log-determinant approximation, arXiv preprint arXiv:2307.02152, (2023)

[4] S. Gaudreault, M. Tokman, and G. Rainwater, Kiops: A fast adaptive krylov subspace solver for exponential integrators, Journal of Computational Physics, 372 (2018), pp. 236–472.

[5] R. A. Meyer, C. Musco, C. Musco, and D. P. Woodruff, Hutch++: Optimal stochastic trace estimation, in Symposium on Simplicity in Algorithms (SOSA), SIAM, 2021, pp. 142–494


