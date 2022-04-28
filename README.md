# Designing Perceptual Puzzles by Differentiating Probabilistic Programs

This repository contains source code to accompany the SIGGRAPH paper [_Designing
Perceptual Puzzles by Differentiating Probabilistic Programs_](https://arxiv.org/abs/2204.12301)
(Chandra, Li, Tenenbaum, and Ragan-Kelley 2022).

> We design new visual illusions by finding "adversarial examples" for
> principled models of human perception — specifically, for probabilistic
> models, which treat vision as Bayesian inference. To perform this search
> efficiently, we design a _differentiable_ probabilistic programming language,
> whose API exposes MCMC inference as a first-class differentiable function. We
> demonstrate our method by automatically creating illusions for three features
> of human vision: color constancy, size constancy, and face perception.

```bibtex
@InProceedings{chandra2022designing,
  title = {Designing Perceptual Puzzles by Differentiating Probabilistic Programs},
  author = {Kartik Chandra and Tzu-Mao Li and Joshua Tenenbaum and Jonathan Ragan-Kelley},
  booktitle = {Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings (SIGGRAPH '22 Conference Proceedings)},
  month = {aug},
  year = {2022},
  doi = {10.1145/3528233.3530715}
}
```

---

**Contents**
- Differentiable Probabilistic Programming (Sec 2)
  - `razor.py`: our differentiable PPL's implementation (Sec 2.2)
  - `reversible.py`: implementation of reversible learning in JAX (Sec 2.3)
  - `Thermometer.ipynb`: concrete implementation of our worked example (Sec 3)
- Applications (Sec 3)
  - `Color constancy.ipynb` (Sec 4.1)
    - `cc/`: pre-rendered masks to be composited by renderer
  - `Size constancy.ipynb` (Sec 4.2)
  - `Face perception.ipynb` (Sec 4.3)
    - `softras.py`: implementation of SoftRas differentiable renderer in JAX
    - `basel_cmplx.npz`: low-poly simplified mesh of the Basel Face Model

**Requirements:**
- Python 3 (3.9.5), ImageMagick
- jax (0.2.20), jaxlib (0.1.71+cuda11)
- Jupyter Lab, numpy, matplotlib, imageio, tqdm
