# Feature contract

The feature contract fixes what one timed call calculates for each adapter.
[`configs/benchmark/main.json`](../configs/benchmark/main.json) holds the
groups and the output counts. [`bench/ibsi_families.py`](../bench/ibsi_families.py)
holds the group members and the IBSI family of each feature code. The timing
rules are in [`timing_method.md`](timing_method.md), and the inputs are in
[`benchmark_design.md`](benchmark_design.md).

## Timed groups

One call calculates all the families that the adapter supports in one group.
All families of a group read the same stored representation. Thus the adapter
can share work inside the group, as its own code does. For example, MIRP
shares its matrix caches between compatible features inside the call.

| Group | IBSI families | Representation | Adapters |
| --- | --- | --- | --- |
| `group_intensity` | Intensity-based statistics | Raw image | All five |
| `group_local_intensity` | Local intensity | Raw image | Four. PyRadiomics has none. |
| `group_morphology` | Morphology without Moran's I and Geary's C (`morphology_shape`) | Raw image and mask | All five |
| `group_spatial_morphology` | Moran's I (`N365`) and Geary's C (`NPT7`) (`morphology_spatial`) | Raw image and mask | Four. PyRadiomics has none. |
| `group_texture` | Intensity histogram, GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM | FBN32 image | All five. PyRadiomics has no GLDZM. |
| `group_ivh` | Intensity-volume histogram | IVH grid (FBS1 or FBN1000) | Four. PyRadiomics has none. |

- A group time covers only the families that the adapter supports. Thus the
  PyRadiomics texture time covers fewer families.
- A group time is not a full-extraction time. Never add group times to make
  one.
- Every group uses 3D calculation with distance 1. GLCM and GLRLM merge their
  directional matrices in 3D (`3d_merge`). The other texture families use one
  3D matrix.
- An adapter that cannot give `3d_merge` for a group gets `unsupported`. No
  native directional average replaces it.
- Moran's I and Geary's C compare every pair of ROI voxels. Thus their cost
  grows with the square of the ROI voxels. They have their own group, so the
  shape group stays comparable across all five adapters.
- Local intensity uses a 1 cm³ sphere (radius 6.2 mm). At 0.5 mm voxels, the
  sphere holds about 8,000 voxels, so its cost grows faster than the voxel
  count. It has its own group for the same reason as Moran's I: one adapter
  omits it, so the intensity group stays comparable across all five adapters.

## Adapter releases and settings

The releases come from [`configs/adapters`](../configs/adapters). Each adapter
runs in its own isolated environment.

| Adapter | Release | Python | Source |
| --- | --- | --- | --- |
| Pictologics | 1.0.0 | 3.12 | PyPI, tag `v1.0.0`, source commit `3cb90e6438ba41fdcffd1a81fbe61d90c4f3ad48` |
| PyRadiomics | 3.1.0 | 3.11 | Official tag commit `6a761c4`. Its distribution metadata reports `3.0.1a1`. |
| MIRP | 2.7.0 | 3.12 | PyPI |
| MEDimage | 0.9.8 | 3.10 | PyPI (`medimage-pkg`), with NumPy 1.26.4 and pandas 1.5.3 pins |
| Z-Rad | 26.8.0 | 3.12 | PyPI (`z-rad`) |

The adapter code is in [`bench/adapters`](../bench/adapters):

- **Pictologics** calls the family code directly and reads the stored levels
  without re-binning.
- **PyRadiomics** enables its complete eligible 3D feature set by class and
  name. The `shape2D` class stays off. `weightingNorm="no_weighting"` sums the
  GLCM and GLRLM direction matrices before the features, which is the IBSI 3D
  merge.
- **PyRadiomics** has no mode without discretisation. The adapter uses a bin
  width of 1 after a check that the ROI levels are integers that start at 1.
  On the stored levels, this is the identity.
- **MIRP** uses `by_slice=False`, exact calculation (`no_approximation=True`)
  and `3d_volume_merge`. Timed tasks turn off the MIRP output filter for IBSI
  features, so all 190 outputs stay.
- **MEDimage** calls its low-level family modules, because it has no stable
  unified extraction interface. The intensity type is `definite`.
- **Z-Rad** uses `aggr_dim="3D"` and `aggr_method="MERG"`. Moran's I and
  Geary's C come from its opt-in `morphology_correlation` group.

## Native output counts

`adapter_inventories` in [`main.json`](../configs/benchmark/main.json) gives
the native outputs of each adapter and group. A zero marks an unsupported
group. The controller records that group as `unsupported` and does not start
it.

| Adapter | `group_intensity` | `group_local_intensity` | `group_morphology` | `group_spatial_morphology` | `group_texture` | `group_ivh` | Native outputs | IBSI Table 2 definitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pictologics | 18 | 2 | 27 | 2 | 118 | 7 | 174 | 172 |
| PyRadiomics | 16 | 0 | 17 | 0 | 77 | 0 | 110 | 108 |
| MIRP | 22 | 2 | 30 | 2 | 119 | 15 | 190 | 172 |
| MEDimage | 18 | 2 | 27 | 2 | 117 | 7 | 173 | 171 |
| Z-Rad | 18 | 2 | 23 | 2 | 118 | 6 | 169 | 167 |

- A measured task must return exactly the count of its adapter and group, and
  every value must be finite. Otherwise the controller fails the task.
- [`bench/benchmark_contract.py`](../bench/benchmark_contract.py) checks that
  the group counts add up to the native output count of each adapter.
- A native output count is not an IBSI coverage count. An adapter can return
  outputs without an IBSI Table 2 identity, or two outputs for one definition.
- A documented exact alias adds an IBSI definition but no calculation.
- The contract requires reports to show runtime beside these counts and the
  IBSI coverage. Never divide runtime by a count.

## Feature surface

[`reproducibility/contracts/adapter_feature_surface.csv`](../reproducibility/contracts/adapter_feature_surface.csv)
lists the native feature names that each group output must contain. The names
come from the IBSI 1 comparison evidence of each adapter. The CSV has one row
for each adapter and family, 56 rows in total. IBSI morphology has two rows:
`morphology_shape` and `morphology_spatial`.

- The column `native_feature_names` holds the names, separated by a comma and
  a space.
- The suffix ` [documented exact alias]` marks a name that also stands for a
  second IBSI definition.
- [`reproducibility/contracts/adapter_feature_surface.json`](../reproducibility/contracts/adapter_feature_surface.json)
  describes the CSV. It holds the contract ID `adapter_feature_surface_ibsi1`,
  the CSV SHA-256, the row count, the generation rule and the checksum of the
  IBSI 1 comparison evidence.
- `main.json` binds the CSV by SHA-256 under `evidence`. The contract loader
  fails when the CSV is missing or differs.

| Family | Pictologics | PyRadiomics | MIRP | MEDimage | Z-Rad |
| --- | ---: | ---: | ---: | ---: | ---: |
| `morphology_shape` | 27 | 14 | 27 | 27 | 23 |
| `morphology_spatial` | 2 | 0 | 2 | 2 | 2 |
| `local_intensity` | 2 | 0 | 2 | 2 | 2 |
| `intensity` | 18 | 15 | 20 | 18 | 18 |
| `histogram` | 23 | 2 | 23 | 23 | 23 |
| `ivh` | 7 | 0 | 7 | 7 | 6 |
| `glcm` | 25 | 25 (2 aliases) | 25 | 25 | 25 |
| `glrlm` | 16 | 16 | 16 | 16 | 16 |
| `glszm` | 16 | 16 | 16 | 16 | 16 |
| `gldzm` | 16 | 0 | 16 | 16 | 16 |
| `ngtdm` | 5 | 5 | 5 | 5 | 5 |
| `ngldm` | 17 | 15 (1 alias) | 17 | 16 | 17 |
| Total names | 174 | 108 | 176 | 173 | 169 |

The MIRP `intensity` row holds 20 names for 18 definitions. It also lists the
MIRP energy and root-mean-square outputs with an offset. The mapping does not
count these two as IBSI features, but the audit requires them in the output.

## PyRadiomics coverage

PyRadiomics 3.1.0 supports 8 of the 11 IBSI families. It has no local
intensity, no IVH and no GLDZM, and no Moran's I or Geary's C.

| IBSI family | Definitions | Direct outputs | Exact aliases | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Morphology | 29 | 14 | 0 | 14/29 |
| Local intensity | 2 | 0 | 0 | 0/2 |
| Intensity-based statistics | 18 | 15 | 0 | 15/18 |
| Intensity histogram | 23 | 2 | 0 | 2/23 |
| Intensity-volume histogram | 5 | 0 | 0 | 0/5 |
| GLCM | 25 | 23 | 2 | 25/25 |
| GLRLM | 16 | 16 | 0 | 16/16 |
| GLSZM | 16 | 16 | 0 | 16/16 |
| GLDZM | 16 | 0 | 0 | 0/16 |
| NGTDM | 5 | 5 | 0 | 5/5 |
| NGLDM | 17 | 14 | 1 | 15/17 |
| Total | 172 | 105 | 3 | 108/172 |

- The timed surface holds 110 native outputs. 105 of them match an IBSI
  Table 2 definition directly.
- Five outputs have no IBSI identity: first-order `TotalEnergy`, GLCM `MCC`,
  and `Maximum2DDiameterSlice`, `Maximum2DDiameterColumn` and
  `Maximum2DDiameterRow`. They stay in the timed work but out of the coverage
  count ([`bench/ibsi_mapping.py`](../bench/ibsi_mapping.py)).
- The three plane-specific diameters do not map to the principal axes or to
  the maximum 3D diameter.
- Deprecated PyRadiomics methods stay off, except `Compactness1`,
  `Compactness2` and `SphericalDisproportion`. The adapter enables these three
  by name, because each one returns its IBSI formula directly.
- The first-order class gives raw statistics and two histogram values. The
  adapter puts entropy and uniformity in the histogram family and all other
  first-order outputs in intensity. Thus no output enters the timed work
  twice.
- The raw first-order statistics never count as histogram features. They use
  the raw voxel values, not the discretised levels.

The three exact aliases reuse a calculated value. They add no calculation and
keep the provenance of their source output. The
[PyRadiomics documentation](https://pyradiomics.readthedocs.io/en/v3.1.0/removedfeatures.html)
proves each identity.

| Source output | Direct IBSI definition | Alias IBSI definition |
| --- | --- | --- |
| `original_glcm_DifferenceAverage` | Difference average (`TF7R`) | Dissimilarity (`8S9J`) |
| `original_glcm_ClusterTendency` | Cluster tendency (`DG8W`) | Sum variance (`OEEB`) |
| `original_firstorder_Uniformity` | Intensity histogram uniformity (`BJ5W`) | NGLDM normalised grey level non-uniformity (`5SPA`) |

PyRadiomics returns no value for the NGLDM dependence count percentage
(`6XV8`). The adapter does not fill in a constant, so this definition does not
count. No definition counts when only adapter code can reconstruct it from
other outputs.

### Kurtosis offset

PyRadiomics returns kurtosis without the IBSI correction of −3, as its own
documentation states. Thus its value is 3 higher than the IBSI value. The
adapter subtracts 3 from `original_firstorder_Kurtosis` after the clock, when
it converts the results. This is the only arithmetic change that an adapter
applies to a returned feature value. The IBSI 2 filter code also translates
two filter conventions: it conjugates the MEDimage Laws responses into the
IBSI frame, and it gives Z-Rad the Gabor width as a full width
([`ibsi_compliance.md`](ibsi_compliance.md#all-voxel-rule)).

## Feature-surface audit

[`scripts/audit_adapter_feature_surface.py`](../scripts/audit_adapter_feature_surface.py)
tests the scheduled feature surface on one stored case. It creates no
benchmark observation and writes no ledger. For each adapter and timed group,
it does these steps:

1. It records an unsupported group as `unsupported_declared` and goes to the
   next group.
2. It selects the stored representation with the same selector as the
   benchmark.
3. It runs the timed adapter code path once, without a warmup.
4. It classifies each returned name as mapped to an IBSI code, excluded or
   unmapped.
5. It checks that every returned value is finite.
6. It checks that every CSV name of the member families is in the group
   output. An alias name can come from any group of the same adapter.

The audit also checks that each adapter reports its configured version. For
PyRadiomics, this version is the metadata version `3.0.1a1`. With
`--host-profile`, it adds the host preflight. It writes
`adapter_feature_surface_audit.json` and `adapter_feature_surface_audit.csv`
with `benchmark_observations_created: false`. It exits with 0 when all groups
pass and with 1 when one fails.

```bash
poetry run python scripts/audit_adapter_feature_surface.py \
  --dataset-dir data/benchmark/pillar1 \
  --case-id p1_reference_noise_m1_n032 \
  --output-dir results/adapter-feature-surface-audit \
  --host-profile configs/benchmark/hosts/mac-m4pro-01.json
```

## IBSI family of each feature code

The timed groups and the morphology split are timing workloads only. Each
IBSI code keeps its IBSI Table 2 family.

- `CODE_TO_FAMILY` in [`bench/ibsi_families.py`](../bench/ibsi_families.py)
  maps each code to its family. It has 174 entries: 172 definitions, with two
  IVH definitions at both 10 and 90 percent.
- Moran's I (`N365`) and Geary's C (`NPT7`) run in `group_spatial_morphology`
  but stay in the morphology family.
- Local intensity runs in `group_local_intensity` and stays in the local
  intensity family.
- The PyRadiomics alias `5SPA` stays an NGLDM definition, although its source
  value comes from the first-order class.
- The coverage counts by family in [`bench/report.py`](../bench/report.py) use
  `CODE_TO_FAMILY`.
- The compliance run maps each result row back to the group whose call made
  it ([`bench/compliance/run.py`](../bench/compliance/run.py)). Thus IBSI 1
  accuracy and timing use the same group calls.
