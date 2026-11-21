# shared — the demo box of examples 10–13

`enclosure.py` defines the 8.4 L box once: slanted back, internal brace, off-centre HPNEW driver on the front, the
ground y = 0; the sealed and vented cabinets (`lp.Cabinet` builders), the driver (`driver()`: modes of
`02_azimuthal_modes`, wet profiles selected on the 2D geometry) and the configured scenes `sealed_scene()`,
`exterior_scene()`, `vented_scene()`. `meshes/` (created on the first run) holds the box meshes and the cavity modes,
shared by the four examples.

Coupled equations (q driver modal amplitudes, P cavity modal pressures, S_kn = ∫ (φ_k·n) ψ_n dS on the wet surface,
R the static correction of the truncated cavity modes):

    (ω_k² + 2iζ_kω_kω − ω²) q_k − ρω² (R q)_k + Σ_n S_kn P_n = F_k
    (ω_n² + iηω_nω − ω²) P_n + ρc²ω² Σ_k S_kn q_k = 0

Simplifications of the demo: the spider, former and coil are acoustically transparent; the pocket under the dust cap
is closed by a rigid lid at the cone foot.
