# Concepts

## The ball

The Poincaré ball of curvature \(-c\) is the open ball of radius \(1/\sqrt{c}\) with the metric tensor \(g_x = \big(\tfrac{2}{1 - c\|x\|^2}\big)^2 g^E\) (the paper's eq. 2 with \(c = 1\)). Distances grow without bound near the boundary, and the circumference of a hyperbolic circle of radius \(r\) grows like \(\sinh r\): a tree with branching factor \(b\) needs room that grows like \(b^{\text{depth}}\), and that is exactly what hyperbolic space supplies (Sarkar 2011; Sala, De Sa, Gu and Ré 2018).

The distance between two points is the paper's eq. 3,

\[
d(x, y) = \operatorname{arcosh}\Big(1 + \frac{2\,\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)}\Big),
\]

implemented as `PoincareBall.distance_arcosh`; the gyrovector form \((2/\sqrt{c})\,\operatorname{artanh}(\sqrt{c}\,\|(-x) \oplus_c y\|)\) is `PoincareBall.distance`. Both agree to \(10^{-8}\) in the test suite. Exponential and logarithmic maps, Möbius addition, the conformal factor, and conversions to and from the Lorentz model are all in [`poincare_memory.geometry`](api.md#geometry).

## Depth and direction

Every memory's point has two independent components.

- **Depth**: the hyperbolic distance from the origin. Roots sit at `base_depth`; every level adds `depth_step`. Depth is read back from the geometry (`Memory.depth(id)`), not from a table, and it is the paper's "distance from the centre encodes hierarchy".
- **Direction**: from the text embedding, L2-normalised, then pulled toward the parent's direction along the geodesic by the fraction `inherit` (default 0.25). Inheritance makes siblings cluster and keeps branches angularly coherent without erasing what each memory is about.

`cone_constrained=True` additionally forces each child inside its parent's [entailment cone](#entailment-cones). That makes cone queries exact for the declared hierarchy but collapses semantic directions onto a few rays, so it is off by default; the explicit hierarchy serves `subtree` and `ancestors` instead.

## Entailment cones

Ganea, Bécigneul and Hofmann (2018) attach to each point \(x\) (outside a small central region of radius \(K\)) a geodesically convex cone opening toward the boundary, with half-aperture \(\psi(x) = \arcsin\big(K(1 - \|x\|^2)/\|x\|\big)\). "\(y\) lies in the cone of \(x\)" is the geometric form of "\(x\) is more general than \(y\)", and cones are transitive. `mode="cone"` returns the memories inside a memory's cone, ranked by score; `cone_energy` gives the violation angle.

## Time

The paper's "time-weighted embeddings" are a pluggable policy mapping a memory's age (and optionally its access history or validity window) to a weight in \((0, 1]\):

| Policy | Weight | From |
|---|---|---|
| `HalfLife(hours)` | \(0.5^{\Delta t / h}\) | the \(0.995^{\text{hours}}\) recency of Generative Agents is `generative_agents_recency()` |
| `Ebbinghaus(stability_hours, boost_per_access)` | \(e^{-\Delta t / S}\), \(S\) grows with accesses | MemoryBank's forgetting curve |
| `ValidityWindow()` | 1 inside `[valid_from, valid_until]`, tiny outside | bi-temporal facts (Graphiti/Zep) |
| `Product(p1, p2, …)` | elementwise product | combine them |

Weights enter the score as \(\text{time\_coef} \cdot \ln w\), so a halving of the weight costs \(\ln 2\), the same as moving \(0.69\) further away in the ball. `search(at_time=…)` evaluates the policy at any moment.

## Scoring

`search` ranks by

\[
\text{score} = -\,d + \text{depth\_bias}\cdot\text{depth} + \text{time\_coef}\cdot\ln w .
\]

### Why the reference shell

With fixed text embeddings, raw hyperbolic distance between a query and the stored points is biased toward shallow memories. Once two directions differ by an angle \(\theta\), the distance between a query at depth \(D_q\) and a memory at depth \(D_m\) is approximately \(D_q + D_m + 2\ln\sin(\theta/2)\): a root at depth 1 beats a leaf at depth 3 unless the leaf's direction is within about 25° of the query's, which off-the-shelf embeddings almost never deliver. Measured on the bundled demo, raw distance returned the root for every query.

`metric="shell"` therefore moves both the query and the memory to the root shell before measuring, so \(d\) is semantic mismatch in hyperbolic units and is comparable across levels. Depth enters explicitly through `depth_bias` (prefer specific memories with \(> 0\), general ones with \(< 0\)) and through the structural modes. This is how the implementation realises the paper's "similarity measures that account for both semantic content and structural relationships".

`metric="hyperbolic"` keeps the raw distance to the memory's actual point and is the right choice when the embedder is itself hyperbolic (for example Hierarchy Transformers), or when generality is what you want.

## Training-free versus learned placement

`Memory.add` places points deterministically from the embedding and the declared parent, which is what an agent needs at write time. For analysis you can also embed a whole hierarchy at once:

- `sarkar_layout(edges)` places a tree exactly by Sarkar's construction (children on a circle of hyperbolic radius \(\tau\) around each node, avoiding the parent's direction), with a float64 safety check `max_safe_depth`;
- `fit_poincare_embeddings(closure)` learns points by Riemannian SGD on the transitive closure (Nickel and Kiela 2017), with `fit_euclidean_embeddings` as the same-loss baseline and `reconstruction()` for MAP and mean rank.

## Any vector database

Map points to the Lorentz model and negate the time coordinate (`to_lorentz`, `to_mips_vectors`): the dot product with a query's Lorentz vector is then exactly its Lorentzian inner product, a decreasing function of hyperbolic distance, so any maximum-inner-product index returns exact hyperbolic nearest neighbours. `Memory.build_index()` does this for the reference shell with a NumPy or FAISS backend; see [Scaling](guides/scaling.md).
