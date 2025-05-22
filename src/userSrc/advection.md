### Concept: Advection-based Scalar Field Transformation

We want to transform a source scalar field (density map) into a target scalar field using **advection** — the transport of quantities through a velocity field. Here's how the method works conceptually and algorithmically.

---

### 1. **What is Advection?**
Advection refers to the process of transporting a quantity (like heat, mass, or density) along a velocity field. In our case, the scalar field is the density, and we seek a velocity field that transports the source density to resemble the target.

Mathematically, the advection equation is:

\[ \frac{\partial f}{\partial t} + \vec{v} \cdot \nabla f = 0 \]

We solve this using **semi-Lagrangian backtracing**, where we ask: for each grid cell in the target, where did this quantity come from in the source?

---

### 2. **Semi-Lagrangian Advection Algorithm**
For each point \( (i,j) \) in the grid:

1. Trace backward along the velocity field: 
   \[ (x', y') = (i, j) - \Delta t \cdot \vec{v}_{i,j} \]

2. Sample the source field at \( (x', y') \) using bilinear interpolation.

3. Store this sampled value in the new (advected) field at \( (i,j) \).

This means we assume the density at a point in the target came from a previous point in the source, and pull that value forward.

---

### 3. **How to Compute the Velocity Field?**
This is the most crucial part if you're matching one field to another.

#### A. Prescribed Field (e.g. radial push or vortex):
Useful for demonstrations. These are hardcoded based on known functions.

#### B. Derived Field from Source & Target:
You want a vector field \( \vec{v} \) such that:
\[ f_{\text{target}}(x) \approx f_{\text{source}}(x - \vec{v}(x)) \]

Approaches:

1. **Displacement Field Matching**:
   \[ \vec{v}(x) = \arg\min_{\vec{v}} \| f_{\text{source}}(x - \vec{v}) - f_{\text{target}}(x) \|^2 \]
   This is nontrivial, often solved via gradient descent or optimization.

2. **Optimal Transport** (Wasserstein distance):
   This computes the minimal cost to transport one density distribution into another, and gives a mass-preserving velocity field. It's ideal for your goal, but computationally heavier.

3. **Image Morphing Techniques**:
   Use optical flow or other registration techniques to compute a smooth displacement between scalar fields.

---

### 4. **Iterative Refinement**
Rather than computing a perfect velocity field in one shot, you can iteratively:

1. Estimate velocity from difference in gradients or fields.
2. Advect source using current velocity.
3. Recalculate velocity field based on residual error.
4. Repeat until loss converges.

---

### Summary
- You advect source densities using a velocity field.
- Velocity can be prescribed or computed from field mismatch.
- Semi-Lagrangian integration is used for stability.
- Best results come from optimal transport or learned velocity fields.
