# Real-data parity fixture

`uci_appliances.csv` is a compact excerpt of the **Appliances Energy
Prediction** dataset donated by Luis Candanedo to the UCI Machine Learning
Repository.

- Source: https://doi.org/10.24432/C5VC8G
- Download: https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Source archive SHA-256:
  `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a`
- Extracted fixture SHA-256:
  `c45620dca460db486b8a40afa5be5be953a29222e286f0d0396c42f2d40e92b8`
- Selection: final 640 chronological observations (2016-05-23 07:30 through
  2016-05-27 18:00), sampled every ten minutes.
- Columns retained unchanged: timestamp, appliance energy, four indoor sensor
  values, and two outdoor weather values.

The fixture is an attributed adaptation permitted by CC BY 4.0. It is used
only to validate inference parity and report forecast metrics; it is not a
claim that the selected future weather measurements would be available in a
live forecast.

`uci_appliances_golden.csv` contains the three frozen target forecasts from
the original reference implementation. `golden_manifest.json` pins the
fixture, checkpoint, config, reference revision, execution semantics, output,
and tolerance hashes. The TimesFM 3.0 checkpoint used to produce these outputs
carries Google's separate non-commercial model-weight license; see the
checkpoint license before reuse.

Regenerate it after downloading the source archive:

```bash
python tests/fixtures/real/generate_uci_appliances.py /path/to/archive.zip
```

The golden is immutable in normal development. If the checkpoint or model
semantics change, establish a separately reviewed oracle outside this project
and update the golden plus every manifest hash together.
