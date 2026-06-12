# mars-ice

## Water simulation ("Warm the planet")

The frontend slider melts predicted ice and floods MOLA topography. Assumptions:

- Zonal annual-mean temperature `T(lat) = 215 K − 55 K · sin²(lat)`. The slider
  sets a **global-mean** warming, applied with polar amplification
  `ΔT(lat) = ΔT_g · (0.8 + 0.6·sin²lat)` (area-weighted mean = 1; poles warm
  1.4×, equator 0.8× — thick-CO₂ Mars GCMs shrink the pole-equator gradient).
- Melt onset at 252 K annual mean, not 273 K: sustained summer melt starts well
  below freezing annual means (Greenland's ablation zone is −10…−20 °C). With
  amplification, melting sweeps equator → poles over ΔT_g ≈ +46…+66 °C; at the
  slider max (+80 °C) the equator sits near +6 °C annual mean. Smoothstepped
  over 5 K. Deep ice (5 m+) lags 15 K (insulation). Pressure is ignored —
  assumes a thickened atmosphere.
- Meltwater per cell = probability × assumed column (4 m shallow, 20 m deep),
  cos-latitude area-weighted, **counting only cells the confidence slider
  keeps** (per-row volumes are binned cumulatively by uncertainty byte, so the
  budget follows the display filter). The polar reservoir — layered deposits +
  circumpolar ice, ≈5×10⁶ km³ ≈ 33 m global-equivalent, the dominant
  inventory — is a 1.5 km slab poleward of ±78°, exempt from the confidence
  filter (observed, not predicted).
- Water level: hypsometric inversion of the meltwater volume over the
  areoid-referenced MEGT grid; every cell below the level renders as water
  (equipotential fill, no flow routing). Disconnected basins fill to the same
  level.
- Sea surface freezes over (rendered slate blue-grey, distinct from the
  white ground-ice overlay) where the warmed annual mean is
  below 271 K — summer melt feeds the basins, but a standing sea's surface
  follows the annual mean, so high-latitude seas are ice-covered until late
  in the warming.
- **Ancient water inventory** toggle: early Mars had ~6× today's accessible
  water (since lost to space / locked in crustal minerals). The toggle adds
  exactly enough water that a full melt reaches the Deuteronilus
  paleo-shoreline at −3,792 m (~128 m global-equivalent total) — the classic
  northern-ocean picture. Released in proportion to overall melt progress
  (it returns as precipitation, not as a polar slab), so it ramps in smoothly
  from the first melt.
- Elevation asset: `web/elevation_rg16.png`, 16-bit metres packed R=high/G=low
  byte (browser canvases are 8-bit/channel), decoded via `web/elevation.json`
  min/max. Exported by `modal run modal_app.py::elevation_main`.

Today's full melt: ≈5.3M km³ (~37 m global layer), sea level ≈ −4,600 m — a
north polar sea plus a deep Hellas, honestly short of the ancient ocean.
Published present-day exchangeable inventory: 20–30 m GEL (caps dominate);
hypsometry of the export says the Deuteronilus shoreline needs 18.5M km³.
