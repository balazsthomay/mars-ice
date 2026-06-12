# mars-ice

## Water simulation ("Warm the planet")

The frontend slider melts predicted ice and floods MOLA topography. Assumptions:

- Zonal annual-mean temperature `T(lat) = 215 K − 55 K · sin²(lat)`; ice melts
  where `T + ΔT` crosses 273.15 K (smoothstepped over 5 K), so melting sweeps
  equator → poles. Deep ice (5 m+) lags by 15 K (insulation). Pressure is
  ignored — assumes a thickened atmosphere.
- Meltwater per cell = probability × assumed column (4 m shallow, 20 m deep),
  cos-latitude area-weighted. Polar layered deposits are approximated as a
  1.2 km slab poleward of ±80° (~19 m global-equivalent — the dominant
  reservoir; the model rasters can't see their thickness).
- Water level: hypsometric inversion of the meltwater volume over the
  areoid-referenced MEGT grid; every cell below the level renders as water
  (equipotential fill, no flow routing). Disconnected basins fill to the same
  level.
- Elevation asset: `web/elevation_rg16.png`, 16-bit metres packed R=high/G=low
  byte (browser canvases are 8-bit/channel), decoded via `web/elevation.json`
  min/max. Exported by `modal run modal_app.py::elevation_main`.

At max warming (+120 °C) everything melts: ≈3.2M km³ (22 m global layer),
sea level ≈ −4,900 m — a north polar sea plus a deep Hellas, honestly short of
the classic Vastitas Borealis ocean (which needs ~100 m global-equivalent).
