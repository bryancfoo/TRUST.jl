# The plasma dispersion function

This document derives everything `src/plasma_dispersion.jl` implements, in the package's own
sign and normalization convention. It is the reference cited by `Z`, `Zprime` and every
susceptibility built on them from Phase 1 onward.

## 1. Definition, the Landau contour, and continuation to all quadrants

The Fried–Conte plasma dispersion function is defined, for `Im ζ > 0`, by

    Z(ζ) = (1/√π) ∫_{-∞}^{∞} e^{−x²} / (x − ζ) dx        (Im ζ > 0),

the integral converging because the pole at `x = ζ` lies off the real integration path. This is
`conventions.md` 1.5's `Z(ξ) = (1/√π) ∫ e^{−x²}/(x−ξ) dx`.

For `Im ζ ≤ 0` the integral as written no longer has a pole-free path, and `Z` is defined by
**analytic continuation** along the Landau contour: deform the integration contour so that it
always passes *below* the pole at `x = ζ`, whichever half-plane `ζ` is in. Concretely:

- **`Im ζ > 0`:** the real axis already passes below the pole (which sits above it); no
  deformation is needed.
- **`Im ζ = 0`:** the contour is indented with a small semicircle *below* the pole, giving the
  principal value plus half a residue: `Z(ξ) = P∫ e^{−x²}/(x−ξ) dx / √π + iπ e^{−ξ²}/√π` (the
  Plemelj form; Appendix A.1.8 derives the general `πi g(z₀)` semicircle residue this is an
  instance of).
- **`Im ζ < 0`:** the contour must duck below the pole, which is now in the lower half-plane, so
  it detours *below* the real axis and back up, picking up the pole's full residue on the way:
  the real-axis integral plus `2πi e^{−ζ²}/√π` (Appendix A.1.10–A.1.11's residue theorem, applied
  to a pole now enclosed by the deformed contour).

This is the **Landau prescription**: physically, it is equivalent to solving an initial-value
problem with time dependence `e^{−iωt}` and letting `Im ω → 0⁺` before `t → ∞`, so that a
mode's causal response is analytic in the upper half of the complex frequency plane and
continued from there. Section 7 (Decision D24) records that this is Sheffield's **Chapter 5**
convention and not his Appendix A.4's.

## 2. `Z(ζ) = i√π erfcx(−iζ)`, and why this is the continuation

The scaled complementary error function is `erfcx(z) = e^{z²} erfc(z)`, and the Faddeeva
function is `w(z) = e^{−z²} erfc(−iz)`. Substituting `z → −iz` shows they are the same function
under the package's argument convention:

    erfcx(−iz) = e^{(−iz)²} erfc(−iz) = e^{−z²} erfc(−iz) = w(z).

`erfc` is entire (it is `1` minus the entire function `erf`), so `w(z) = e^{-z^2}\mathrm{erfc}(-iz)`
is entire — it has **no poles anywhere in the complex plane**. The claim

    Z(ζ) = i√π · w(ζ) = i√π · erfcx(−iζ)

is a classical identity (Fried & Conte 1961): it agrees with Section 1's integral definition on
`Im ζ > 0` (both solve the same defining ODE with the same boundary behavior as `Re ζ → ±∞`,
Section 3 below), and because `i√π w(ζ)` is entire, it is automatically the **unique** analytic
continuation of that integral to the rest of the plane — exactly what the Landau-contour
prescription of Section 1 constructs by hand. This is why `src/plasma_dispersion.jl` needs no
case split on the sign of `Im ζ`: one entire formula covers every quadrant, and `erfcx` supplies
it directly (`SpecialFunctions.erfcx`, which accepts a `Complex` argument).

## 3. The derivatives `Z'` and `Z''`

Differentiate the integral definition under the integral sign (valid for `Im ζ > 0`, then
continued): `d/dζ [1/(x−ζ)] = 1/(x−ζ)²`, so

    √π Z'(ζ) = ∫ e^{−x²}/(x−ζ)² dx.

Separately, integrate `∫ (−2x e^{−x²})/(x−ζ) dx` by parts, using `d(e^{−x²})/dx = −2x e^{−x²}`
and `d/dx[1/(x−ζ)] = −1/(x−ζ)²`:

    ∫ (−2x e^{−x²})/(x−ζ) dx = [e^{−x²}/(x−ζ)]_{−∞}^{∞} + ∫ e^{−x²}/(x−ζ)² dx = √π Z'(ζ),

the boundary term vanishing because `e^{−x²} → 0`. But the same integral can be split using
`−2x = −2(x−ζ) − 2ζ`:

    ∫ (−2x e^{−x²})/(x−ζ) dx = −2∫ e^{−x²} dx − 2ζ ∫ e^{−x²}/(x−ζ) dx = −2√π − 2ζ√π Z(ζ).

Equating the two expressions for the same integral and dividing by `√π`:

    Z'(ζ) = −2(1 + ζ Z(ζ)).                                                    (★)

This is implemented directly, never by differencing (Decision D6). Differentiating (★) once
more with the product rule gives the second derivative needed by the dual and `ChainRulesCore`
methods of `Zprime` (Decision D7, D8):

    Z''(ζ) = −2(Z(ζ) + ζ Z'(ζ)).

## 4. Relation to Sheffield's Rw and Iw

Sheffield writes the Maxwellian susceptibility (Eq. 5.2.5) as
`χ_e = −(α²/2) Z'(x_e) = α²[Rw(x_e) + i Iw(x_e)]`, so by matching real and imaginary parts,

    Rw(x) = −Re Z'(x) / 2,      Iw(x) = −Im Z'(x) / 2                          (Eqs. 5.2.7, 5.2.8)

For real `x`, write `Z(x) = −2D(x) + i√π e^{−x²}` with `D` the Dawson function
`D(x) = e^{−x²}∫_0^x e^{p²} dp` (the standard real-axis form of `Z`, itself a consequence of
Section 2: `erfc(−ix)` for real `x` reduces to `1 + (2i/√π)∫_0^x e^{p²}dp`, giving
`w(x) = e^{−x²} + (2i/√π) D(x) e^{... }`; equivalently, this is the real-axis value tabulated by
Fried & Conte). From (★) with `x` real,

    Re Z'(x) = −2(1 + x·Re Z(x)) = −2(1 − 2x D(x)),
    Im Z'(x) = −2 x·Im Z(x)      = −2√π x e^{−x²},

so

    Rw(x) = 1 − 2x D(x),      Iw(x) = √π x e^{−x²} > 0 for x > 0.               (Eqs. 5.2.7, 5.2.8)

`Rw(x) = 1 − 2x D(x)` is Sheffield's own `1 − 2x e^{−x²}∫_0^x e^{p²}dp` rewritten using
`D(x) ≡ e^{−x²}∫_0^x e^{p²}dp`. `Iw(x) > 0` for `x > 0` is the Landau-damping sign and the
fastest check that a continuation is not conjugated (Section 7).

## 5. The small- and large-`x` expansions

**Small `x`** (Sheffield Eq. 5.2.9): the Dawson function's Taylor series is
`D(x) = x − (2/3)x³ + (4/15)x⁵ − …`, so

    Rw(x) = 1 − 2x D(x) = 1 − 2x² [1 − (2/3)x² + (4/15)x⁴ − …].

**Large `x`** (Sheffield Eq. 5.2.10): the Dawson function's asymptotic series is
`D(x) ~ 1/(2x) + 1/(4x³) + 3/(8x⁵) + …`, so `2x D(x) ~ 1 + 1/(2x²) + 3/(4x⁴) + …` and

    Rw(x) ~ −1/(2x²) − 3/(4x⁴) − 15/(8x⁶) − … = −(1/2x²) [1 + 3/(2x²) + 15/(4x⁴) + …].

This series is asymptotic (divergent), not convergent: the error after truncation is bounded by
the first omitted term only once `x` is large enough for the terms to be decreasing, which is
why the Step 0.2 test (T6) excludes `x = 3` and its tolerance is stated as a multiple of the
first omitted term rather than as a fixed power of `x`.

## 6. The reflection formula

The Faddeeva function obeys `w(−z) = 2e^{−z²} − w(z)` (a standard property, itself following
from `erfc(z) + erfc(−z) = 2`). Multiplying by `i√π`:

    Z(−ζ) = i√π w(−ζ) = i√π(2e^{−ζ²} − w(ζ)) = 2i√π e^{−ζ²} − Z(ζ),

i.e.

    Z(ζ) + Z(−ζ) = 2i√π e^{−ζ²}                                                (reflection)

for every complex `ζ`, not just the real axis. This is how `test/test_plasma_dispersion.jl`
gets a lower-half-plane reference (`Z_ref_lower`) from an upper-half-plane quadrature oracle
(`Z_ref_upper`) without ever evaluating the defining integral off its convergent half-plane:
`Z(ζ) = 2i√π e^{−ζ²} − Z(−ζ)`, and `−ζ` is in the upper half-plane whenever `ζ` is in the lower
one.

## 7. Sheffield's two time conventions (Decision D24)

The package's convention — the Landau prescription of Section 1, contour below the pole,
equivalent to `e^{−iωt}` with `ω` given a small positive imaginary part, `𝒞φ(ξ)` and `Z(ξ)`
defined for `Im ξ > 0` and continued — matches Sheffield **Chapter 5** (Eqs. 5.2.5–5.2.8, where
`Iw(x) > 0` for `x > 0`, Section 4 above) and the standard Fried–Conte function.

**Sheffield's Appendix A.4 uses the opposite time convention.** Its Eq. A.4.3 assumes solutions
of the form `F(x, v, t) = F_0(v) + F_1(v)e^{−ikx+iωt}` — note `+iωt`, not `−iωt` — and its
Fourier–Laplace transform (Eq. A.4.4) is defined with `e^{−(iω+γ)t}` and stated to converge for
`Im ω < 0` (the text below Eq. A.4.2: "if the imaginary part of ω is negative... the system is
unstable"), the opposite of the Chapter 5/standard-Z convergence half-plane. Correspondingly, in
Eq. A.4.9 the pole of the integrand sits at `v = (ω − iγ)/k`, which is *below* the real velocity
axis for the small positive `γ` used to select the causal (growing-mode) solution — the mirror
image of the Landau contour used here. A reader who takes the *contour construction* from
Appendix A and the *sign convention* from Chapter 5 will get a `Z` that is conjugated relative to
the one this package implements.

**The one-line check that the continuation used here is the right one:** on the real axis,

    Im Z(x) = +√π e^{−x²} > 0.

If a derivation or a test ever produces the opposite sign, the contour has been taken from the
wrong section of Sheffield.
