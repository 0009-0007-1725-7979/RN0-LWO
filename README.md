## Architecturally Aware Optimisation for Custom GNN-based Models in Communication Networks
#### ABSTRACT:
Custom GNN architectures have emerged as the state-of-the-art approach for modeling network performance in communication networks, outperforming pure GNN designs that struggle to capture complex path-dependent behaviors. However, while architecture design has received significant attention in recent research, the optimization of such heterogeneous models remains largely unexplored. 

This study argues that architectural heterogeneity introduces optimization asymmetry, necessitating layer-wise balancing mechanisms for structural coordination. Through theoretical analysis and empirical evidence, this study reveals how hybrid models composed of structurally diverse components exhibit uneven optimization dynamics, and therefore benefit from layer-wise optimization. We reinterpret the trust-ratio mechanism from LAMB to explore adaptation rates across heterogeneous layers. We formalize the grad-to-weight ratio and its coefficient of variation as intrinsic diagnostics for characterizing convergence behaviour. Through controlled experiments on network performance prediction tasks, we show that layer-wise adaptive optimization yields more stable convergence, achieving up to 82% reduction in post-convergence loss volatility and improved robustness relative to standard Adam.


#### keywords:
Broadband Telecommunications, Network Modeling, Graph Neural Network (GNN), Layer-wise Optimisation
