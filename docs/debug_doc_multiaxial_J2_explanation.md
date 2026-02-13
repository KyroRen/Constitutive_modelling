## Explanation of the “strange” σ11–ε11 curve under multiaxial loading

This note explains why, in the multiaxial Chaboche test with the stress path

- σ11: 0 → 400 → 200 → 0 MPa  
- σ31: 0 → 400 → 700 → 0 MPa

the σ11–ε11 curve shows continued plastic accumulation during the segment where
σ11 decreases from 400 to 200 MPa. At first sight this looks like “unloading
with plastic strain growth”, which can feel counter‑intuitive.

### 1. What drives plasticity in J2 theory

In classical associated J2 plasticity, the yield function is

\[
f(\boldsymbol{\sigma}) = \sqrt{\frac{3}{2}\,\boldsymbol{s}:\boldsymbol{s}} - \sigma_y,
\]

where \(\boldsymbol{s}\) is the deviatoric stress. Plastic flow occurs when
\(f \ge 0\) and the plastic multiplier increment \(dp > 0\).

The flow rule is

\[
d\boldsymbol{\varepsilon}^p = d\lambda\,\frac{\partial f}{\partial \boldsymbol{\sigma}}
          = dp\,\boldsymbol{n},
\]

where \(\boldsymbol{n}\) is the *direction* of the deviatoric stress

\[
\boldsymbol{n}
  = \frac{3}{2}\,\frac{\boldsymbol{s}}{\sqrt{\frac{3}{2}\,\boldsymbol{s}:\boldsymbol{s}}}.
\]

Two key points:

1. Plasticity is driven by the *equivalent* (von Mises) stress, not by any
   single stress component.
2. The direction of the plastic strain increment is given by \(\boldsymbol{n}\),
   i.e. by the **direction of the deviatoric stress**, not simply by the sign
   of σ11.

The component increments are

\[
d\varepsilon^p_{ij} = dp\,n_{ij}.
\]

As long as \(dp > 0\) and \(n_{11} \ne 0\), we will have
\(d\varepsilon^p_{11} \ne 0\), regardless of whether σ11 itself is increasing
or decreasing.

### 2. Pure shear vs combined normal + shear

It is useful to contrast two extreme cases:

#### (a) Pure shear

If we apply *ideal* pure shear, e.g. σ12 ≠ 0 with all normal components zero,
then the deviatoric stress has only shear components and \(\boldsymbol{n}\) has
only shear components. In that case

- \(d\varepsilon^p_{11} = d\varepsilon^p_{22} = d\varepsilon^p_{33} = 0\),
- only the shear components of \(d\boldsymbol{\varepsilon}^p\) are non‑zero.

So in pure shear, J2 plasticity does **not** create normal plastic strain.

#### (b) Combined σ11 and σ31 as in the test

In the present path, both σ11 and σ31 = σ13 are non‑zero. This means that
the deviatoric stress has non‑zero components in both the 11 and 31 directions,
and so does the flow direction \(\boldsymbol{n}\).

During the segment from

- A: σ11 ≈ 400 MPa, σ31 ≈ 400 MPa  
to
- B: σ11 ≈ 200 MPa, σ31 ≈ 700 MPa,

the von Mises equivalent stress (see the σeq–p plot) actually **increases** and
remains well above the yield stress. Therefore:

- The plastic multiplier increment \(dp\) is still positive in this segment;
- The flow direction \(\boldsymbol{n}\) still has a non‑zero 11‑component;
- Consequently, \(d\varepsilon^p_{11} = dp\,n_{11} > 0\).

So from the J2 point of view, this segment A→B is **not unloading** at all:
it is a continuation of plastic loading, with the stress state being rotated
and intensified towards shear rather than relaxed.

### 3. Why the σ11–ε11 curve looks “strange”

If one looks only at σ11, the path

\[
400 \to 200 \text{ MPa}
\]

resembles unloading in a uniaxial test, where one would expect mostly elastic
response and very little new plastic strain. In pure uniaxial loading that
intuition is correct: as σ11 decreases and the equivalent stress moves inside
or towards the yield surface, plastic flow ceases and ε11 follows an almost
linear elastic path.

However, in the present *multiaxial* path:

- While σ11 is decreasing from 400 to 200 MPa, σ31 is simultaneously increased
  from 400 to 700 MPa.
- This causes the von Mises equivalent stress to increase and stay in the
  plastic regime.
- The flow direction has a non‑zero normal component, so the equivalent
  plastic strain \(p\) and the normal plastic strain component ε11^p both
  continue to grow.

As a result, each “cycle” shows:

- σ11 going up to about 400 MPa,
- then dropping to 200 MPa while ε11 *still accumulates plastic strain*,
- leading to a family of σ11–ε11 loops that are shifted progressively to
  the right (ratcheting in the ε11 direction).

This is not a numerical artefact but a natural outcome of J2 plasticity under
non‑proportional combined loading with a significant shear component.

### 4. How to see a more “intuitive” unloading path

If you want σ11–ε11 to look more like classical uniaxial unloading, you can
modify the stress path in the test script, for example:

- Keep σ31 constant while reducing σ11 from 400 to 200 MPa, or  
- Set σ31 = 0 and perform symmetric uniaxial stress cycles.

In those cases, the equivalent stress during the “400 → 200” segment will
decrease significantly, \(dp\) will be small or zero, and the σ11–ε11 path
will show the expected almost‑elastic unloading behaviour.

