# LSHiForest: A Generic Framework for Fast Tree Isolation Based Ensemble Anomaly Analysis

## What This Paper Is About

Anomaly detection (outlier detection) is crucial for big data analytics, but large volume and high dimensionality make traditional algorithms too slow. The tree isolation mechanism, used in methods like **iForest** and **SCiForest**, isolates data points by recursively partitioning the space and scores anomalies based on how quickly they are separated. These methods are fast (logarithmic time) but are limited to specific distance measures (Manhattan, angular) and cannot be easily extended to other metrics or data types.

This paper proposes **LSHiForest**, a generic framework that generalises tree isolation using **Locality-Sensitive Hashing (LSH) forests**. By leveraging LSH families, the isolation mechanism can be applied to *any* distance/similarity measure, data space, or data type where an LSH family exists. The framework is instantiated with several LSH families (ℓ₁, ℓ₂, angular, kernelised), and it is formally shown that iForest and SCiForest are special cases. Experiments demonstrate high detection quality, efficiency (still logarithmic), and versatility.

## How the Framework Works

### Core Idea
The key insight is that an **LSH tree** (built from a subsample) acts as an isolation tree: each data instance is isolated into its own leaf node. The path length from root to leaf, combined with the neighbourhood information preserved by the LSH family, gives the anomaly score. Because LSH families exist for many distances and spaces (e.g., Euclidean, angular, kernelised, set-based), the isolation mechanism becomes generic.

### LSH Forest Background
An LSH family consists of hash functions that map “nearby” points to the same bucket with high probability, and far-apart points to the same bucket with low probability. By concatenating multiple hash functions (increasing α), precision improves; by using multiple hash tables (β), recall is boosted. An **LSH forest** uses variable‑length combined keys, building a prefix tree (trie) where each node applies a hash function from the family. The tree naturally isolates points.

### LSHiForest Architecture
1. **Training:** For each ensemble component, draw a variable‑sized subsample (size ~64–1024). Build an LSH isolation tree by recursively splitting the subsample using random hash functions from a chosen LSH family. A height limit prevents infinitely long paths for identical/very close points. The resulting forest of LSH trees is stored.
2. **Evaluation:** For a test point, traverse each tree to compute its path length. The path length combines information from both the *uncompressed digital trie* and the *compressed PATRICIA trie*. After normalisation, the length is transformed into an anomaly score ∈ (0,1] via \(2^{-h/\mu}\). Scores are averaged across the ensemble.

### Key Components and Formulas

- **Variable subsampling:** Sample size \(s\) drawn uniformly from \(2^{s_{\min}}\) to \(2^{s_{\max}}\) (where \(s\) is chosen uniformly from [6,10], yielding sizes ~64–1024). This gives uniformly distributed tree heights (diversity) and expected size ≈346, improving efficiency.
- **Height limit** \(H(\psi)\) – derived from the average height of a digital trie:  
  \[
  E(H) \approx \frac{2\ln(\psi)}{\ln(v)} + \frac{\gamma - \ln(2)}{\ln(v)} +1 \;\leq\; 2\log_2(\psi) + 0.8327,
  \]
  where \(v\) is the branching factor (number of possible hash values), \(\gamma \approx 0.5772\).
- **Path length computation** (Algorithm 4):  
  For a point \(x\), traverse the tree from the root. At each internal node, compute the hash value \(f_I(x)\) and follow the corresponding child. If the node is a leaf or the depth limit \(L\) is reached, the path length is adjusted.  
  The combined path length is  
  \[
  h(x) = h_c \left(\frac{h_u}{h_c}\right)^{\eta},
  \]
  where \(h_c\) is the path length in the compressed (PATRICIA) trie, \(h_u\) is the path length in the uncompressed digital trie, and \(\eta \in [0,1]\) controls granularity. \(\eta=1\) gives finest isolation for global anomalies; \(\eta<1\) helps for local anomalies.
- **Reference normalisation** \(\mu(\psi)\) – average successful search length in a PATRICIA trie, under equiprobable branching:  
  \[
  \mu(\psi) = 
  \begin{cases}
  \frac{\ln(\psi) + \ln(v-1) + \gamma}{\ln(v)} - \frac12, & \psi > v \\
  1, & 1 < \psi \leq v \\
  0, & \text{otherwise.}
  \end{cases}
  \]
  The branching factor \(v\) is estimated as the average number of distinct hash keys observed per node during training.
- **Anomaly score:**  
  \[
  AS_x = \frac{1}{t} \sum_{i=1}^{t} 2^{-h_i(x)/\mu(v_i)},
  \]
  where \(t\) is the number of trees. The nonlinear scaling \(2^{-x}\) is applied *before* averaging, increasing ensemble diversity.

### How Existing Methods Are Special Cases

- **iForest** uses random axis‑parallel splits. For a chosen attribute, the split function is \(f_i(x_i) = \mathrm{sgn}(x_i - \omega_i)\). The probability that two points fall into the same partition is \(1 - |x_i - y_i|\). Over attributes, the probability becomes \(1 - \frac{1}{m}\sum|x_i - y_i|\). This is an LSH family for the \(\ell_1\) distance. Thus iForest = LSHiForest with an \(\ell_1\)-LSH family.
- **SCiForest** uses hyperplanes with coefficients randomly chosen from \([-1,1]\) and an optimised split point. It corresponds to an angular distance LSH, but with a cumbersome parameter selection; it can be seen as a learning‑based hashing variant.

### Instantiations with Different LSH Families

The paper demonstrates three LSH families beyond the ones implicit in iForest/SCiForest:

1. **\(\ell_p\) (p=1,2) LSH** – uses p‑stable distributions:  
   \[
   f_{\omega,\omega_0}(x) = \left\lfloor \frac{\omega\cdot x + \omega_0}{W} \right\rfloor,
   \]
   where \(\omega\) components are drawn from the Cauchy (p=1) or Gaussian (p=2) distribution, \(\omega_0\sim U[0,W]\), and \(W\) is a bucket size parameter. This produces multi‑fork trees, not only binary.
2. **Angle‑based LSH** – uses \(f_\omega(x) = \mathrm{sgn}(\omega^T x)\) with \(\omega_i\sim \mathcal{N}(0,1)\), enabling similarity in terms of angular distance.
3. **Kernelised LSH** – operates in a kernel‑induced feature space, making local anomalies (in original space) become global anomalies and easier to isolate. The hash function:  
   \[
   f(\phi(x)) = \mathrm{sgn}\!\left(\sum_{i=1}^{\lambda} \omega(i)\, \kappa(x, \hat{x}_i)\right),
   \]
   where \(\omega = \bar{K}^{-1/2} e_\xi\), \(\bar{K}\) is the centred kernel matrix of a sample of size \(\lambda\), and \(e_\xi\) selects \(\xi\) of the \(\lambda\) points. This yields multi‑fork trees if multiple random projections are concatenated.

### Time Complexity

All instances have **logarithmic evaluation** w.r.t. the subsample size \(\psi\), making them orders of magnitude faster than nearest‑neighbour based ensembles (which have linear evaluation cost). For ℓ₁/ℓ₂ and angular LSH, training is \(\Theta(t\psi\log_v(\psi)m)\) and prediction \(\Theta(t\log_v(\psi)m)\). Kernelised LSH incurs an extra \(\lambda^2 m + \lambda^3\) cost for setting up the kernel matrix, and the prediction cost includes a \(\lambda m\) projection step.

## Step‑by‑Step Algorithm Explanation

### Training: Building an LSH Forest (Algorithm 1)
1. For each tree \(i\) from 1 to \(t\):
   - Draw a subsample \(S_i \subseteq X\) using variable subsampling (size \(s_i\) from a distribution that yields ~64–1024 points).
   - Compute the height limit \(H_i\) using equation (1) with an estimated branching factor.
   - Call `LSHiTree` (Algorithm 2) to build a tree from \(S_i\).
2. Return the forest \(\{T_i\}\).

### LSHiTree (Algorithm 2) – Recursive Construction
1. **Base cases:**
   - If the current set \(S\) is empty, return `NULL`.
   - If \(|S|=1\) or the current tree depth index \(I > H\), create a leaf node storing the size and hash index.
2. **Splitting:**
   - Use a randomly chosen hash function \(f_I\) from the LSH family to map each point to a key \(K_j\).
   - If all points map to a **single key** and \(I \le H\), increment \(I\) and re‑hash with a new function (compressing single‑branch paths).
   - If after trying multiple functions still only one key, or \(I>H\), return a node with size \(|S|\).
3. **Recursion:**
   - For each distinct key \(K_i\), recursively build a subtree on the subset \(S_i\).
   - Return an internal node with the hash index and children mapping.

### Prediction: Computing Anomaly Scores (Algorithm 3)
1. For each test point \(x\):
   - Initialise \(AS_x \leftarrow 0\).
   - For each tree \(T_i\) in the forest:
     * Compute path length \(h_i(x)\) by calling `path_length` (Algorithm 4).
     * Normalise: \(AS_x \leftarrow AS_x + 2^{-h_i(x)/\mu(v_i)}\).
   - Average: \(AS_x \leftarrow AS_x / t\).
2. Return all scores.

### Path Length Calculation (Algorithm 4)
Given a node, the current (compressed) depth \(I_{cur}\), and parameters \(\eta\) and \(L\):
- If the node is `NULL` → return \(-1\).
- If the node is a leaf (no children) **or** \(I_{cur} > L\) (depth limit):
  * Return \(I_{cur} \left(\frac{node.Hash\_Index}{I_{cur}}\right)^{\eta} + \mu(node.Size)\).  
    This mixes compressed depth (\(I_{cur}\)) with an approximation of uncompressed depth via the ratio \(node.Hash\_Index / I_{cur}\).
- Else (internal node):
  * Compute the hash key \(K = f_{Hash\_Index}(x)\).
  * If a child with key \(K\) exists: recurse with that child, \(I_{cur}+1\).
  * If no matching child (point is dissimilar): return \((I_{cur}+1)\left(\frac{node.Hash\_Index +1}{I_{cur}}\right)^{\eta}\).  
    This early termination gives short path lengths for outliers that land in previously unseen buckets.

The mix of compressed/uncompressed path length, plus early stopping, gives LSHiForest strong isolation power.

## How Are the Results

The paper evaluates four instantiations (ALSH, L1SH, L2SH, KLSH) against state‑of‑the‑art ensemble methods: iForest (ISO), SCiForest (SCI), EnKNN (average \(k\)-NN ensemble), EnLOF (LOF ensemble), and iNNE. Experiments use both synthetic and 20 real‑world UCI benchmark datasets.

### Key Findings

- **Effect of branching factor \(v\):** On the Twospirals dataset, higher branching (up to \(u=4\)) yields tighter boundaries around dense regions and better detection of surrounded anomalies. Over‑branching can mask anomalies in dense clusters, so \(u\) between 2 and 5 is a good trade‑off.
- **Surrounded anomalies:** L1SH, L2SH, and KLSH achieve high AUC and tight boundaries, while iForest fails (axis‑parallel contours) and SCiForest partially succeeds. EnKNN/EnLOF have broad boundaries.
- **Local anomalies:** On a single‑anomaly scenario with varying locality ratio \(R\), KLSH and SCiForest can detect truly local anomalies (\(R<1\)), while L1SH/L2SH with \(\eta=0\) also show improved capability. EnLOF and iNNE detect them but are much slower.
- **Real‑world benchmarks (20 datasets):**
  - **EnKNN and L2SH** are the most robust, achieving top AUC on many datasets.
  - **L1SH, KLSH, and SCI** also perform well, but SCI can occasionally fail badly (e.g., Power dataset) due to its data‑dependent optimisation.
  - **iForest (ISO) and ALSH** are less stable; iForest often underperforms on complex anomalies, ALSH is unstable.
  - **L2SH** emerges as the overall best trade‑off between detection quality, robustness, and speed.
- **Execution time:** All LSH‑based instances (ALSH, L1SH, L2SH) run about **two orders of magnitude faster** than EnKNN/EnLOF/iNNE. iForest and these instances have similar speed. SCiForest is slower due to optimisation, but still fast at test time. KLSH can be slower as \(\lambda\) grows with \(n\), but still logarithmic in \(\psi\).

Tables from the paper (simplified):
- AUC of L2SH is highest or near‑highest across most datasets (often above 90%).
- L1SH and KLSH are close behind.
- Execution times: e.g., on dataset with 10k points, L2SH finishes in ~6s while EnKNN takes >200s.

## Key Takeaways

1. **LSH forests generalise tree isolation anomaly detection.** Any LSH family can be plugged in, enabling the fast, logarithmic‑time isolation mechanism to work with arbitrary distance measures, data spaces (e.g., kernelised), and data types (e.g., categorical via min‑hash).

2. **iForest and SCiForest are special cases.** iForest is equivalent to an \(\ell_1\)-LSH family, SCiForest to a constrained angular‑LSH variant. The framework reveals their hidden reliance on distance measures, contradicting earlier claims.

3. **Path length mixing and early stopping enhance detection.** The combined compressed/uncompressed path length (with control parameter \(\eta\)) and the ability to stop at internal nodes when no matching hash bucket exists improve sensitivity to both global and local anomalies.

4. **No single LSH family always wins; diversity is powerful.** L2SH (Euclidean) is the most robust overall, but the best instantiation depends on the data and anomaly type. The generic framework allows users to choose the appropriate distance for their domain.

5. **Ensemble with variable subsampling improves robustness.** Using subsample sizes uniformly distributed in log‑scale (64–1024) creates tree‑height diversity, which reduces variance and eliminates the need to tune base‑detector parameters (like \(k\) in \(k\)-NN).

6. **Kernelised LSH can detect local anomalies at lower cost than density‑based methods.** By mapping data to a kernel space, local anomalies become global and can be isolated with logarithmic complexity, avoiding the linear cost of LOF or iNNE.

7. **The framework supports early depth‑limiting** (\(L\) parameter) to detect clustered anomalies (like iForest’s height limit) and provides theoretical estimates for tree height and reference path length, making it practically parameter‑free (except for \(L\) and \(\eta\), which have intuitive defaults).

In summary, LSHiForest retains the speed of isolation‑based methods while achieving state‑of‑the‑art detection quality and remarkable flexibility across data types and distance measures. It is a foundational step toward universal, scalable anomaly detection.
