from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from bench.ibsi_codes import CODE_TO_NAME
from bench.ibsi_families import CODE_TO_FAMILY


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


NAME_TO_CODES: Dict[str, list[str]] = {}
for _code, _name in CODE_TO_NAME.items():
    NAME_TO_CODES.setdefault(_normalize(_name), []).append(_code)


def _code_from_name_in_family(name: str, family: Optional[str]) -> Optional[str]:
    norm = _normalize(name)
    codes = NAME_TO_CODES.get(norm, [])
    if not codes:
        return None
    if family:
        fam_codes = [c for c in codes if CODE_TO_FAMILY.get(c) == family]
        return sorted(fam_codes)[0] if fam_codes else None
    return sorted(codes)[0]


def map_pictologics(feature_name: str) -> Optional[str]:
    # Pictologics encodes IBSI codes as suffixes, sometimes with extra token (e.g., BC2M_10)
    parts = feature_name.split("_")
    # try longest suffix first (3 -> 1)
    for n in range(3, 0, -1):
        if len(parts) >= n:
            code = "_".join(parts[-n:])
            if code in CODE_TO_NAME:
                return code
    return None


PYRADIOMICS_MAP = {
    ("shape", "MeshVolume"): "Volume",
    ("shape", "VoxelVolume"): "Volume voxel counting",
    ("shape", "SurfaceArea"): "Surface area",
    ("shape", "SurfaceVolumeRatio"): "Surface to volume ratio",
    # These three upstream methods are marked deprecated because they are
    # algebraically redundant with Sphericity, but PyRadiomics still
    # calculates and returns them when they are explicitly enabled by name.
    ("shape", "Compactness1"): "Compactness 1",
    ("shape", "Compactness2"): "Compactness 2",
    ("shape", "SphericalDisproportion"): "Spherical disproportion",
    ("shape", "Sphericity"): "Sphericity",
    ("shape", "Maximum3DDiameter"): "Maximum 3D diameter",
    ("shape", "MajorAxisLength"): "Major axis length",
    ("shape", "MinorAxisLength"): "Minor axis length",
    ("shape", "LeastAxisLength"): "Least axis length",
    ("shape", "Elongation"): "Elongation",
    ("shape", "Flatness"): "Flatness",
    ("firstorder", "Mean"): "Mean intensity",
    ("firstorder", "Variance"): "Intensity variance",
    ("firstorder", "Skewness"): "Intensity skewness",
    ("firstorder", "Kurtosis"): "Intensity kurtosis",
    ("firstorder", "Median"): "Median intensity",
    ("firstorder", "Minimum"): "Minimum intensity",
    ("firstorder", "Maximum"): "Maximum intensity",
    ("firstorder", "Range"): "Intensity range",
    ("firstorder", "InterquartileRange"): "Intensity interquartile range",
    ("firstorder", "MeanAbsoluteDeviation"): "Intensity Mean absolute deviation",
    (
        "firstorder",
        "RobustMeanAbsoluteDeviation",
    ): "Intensity Robust mean absolute deviation",
    ("firstorder", "Energy"): "Intensity energy",
    ("firstorder", "RootMeanSquared"): "Root mean square intensity",
    ("firstorder", "Entropy"): "Discretised intensity entropy",
    ("firstorder", "Uniformity"): "Discretised intensity uniformity",
    ("firstorder", "10Percentile"): "10th intensity percentile",
    ("firstorder", "90Percentile"): "90th intensity percentile",
    ("glcm", "JointAverage"): "Joint average",
    ("glcm", "JointEntropy"): "Joint entropy",
    ("glcm", "JointEnergy"): "Angular second moment",
    ("glcm", "MaximumProbability"): "Joint maximum",
    # PyRadiomics names the IBSI Joint Variance implementation ``SumSquares``.
    # Its upstream docstring explicitly identifies this feature as Joint
    # Variance; ``JointVariance`` is not a PyRadiomics feature name.
    ("glcm", "SumSquares"): "Joint variance",
    ("glcm", "SumAverage"): "Sum average",
    ("glcm", "SumEntropy"): "Sum entropy",
    ("glcm", "DifferenceAverage"): "Difference average",
    ("glcm", "DifferenceVariance"): "Difference variance",
    ("glcm", "DifferenceEntropy"): "Difference entropy",
    ("glcm", "Contrast"): "Contrast",
    ("glcm", "Id"): "Inverse difference",
    ("glcm", "Idn"): "Normalised inverse difference",
    ("glcm", "Idm"): "Inverse difference moment",
    ("glcm", "Idmn"): "Normalised inverse difference moment",
    ("glcm", "InverseVariance"): "Inverse variance",
    ("glcm", "Correlation"): "Correlation",
    ("glcm", "Autocorrelation"): "Autocorrelation",
    ("glcm", "ClusterTendency"): "Cluster tendency",
    ("glcm", "ClusterShade"): "Cluster shade",
    ("glcm", "ClusterProminence"): "Cluster prominence",
    ("glcm", "Imc1"): "Information correlation 1",
    ("glcm", "Imc2"): "Information correlation 2",
    ("glrlm", "ShortRunEmphasis"): "Short runs emphasis",
    ("glrlm", "LongRunEmphasis"): "Long runs emphasis",
    ("glrlm", "LowGrayLevelRunEmphasis"): "Low grey level run emphasis",
    ("glrlm", "HighGrayLevelRunEmphasis"): "High grey level run emphasis",
    ("glrlm", "ShortRunLowGrayLevelEmphasis"): "Short run low grey level emphasis",
    ("glrlm", "ShortRunHighGrayLevelEmphasis"): "Short run high grey level emphasis",
    ("glrlm", "LongRunLowGrayLevelEmphasis"): "Long run low grey level emphasis",
    ("glrlm", "LongRunHighGrayLevelEmphasis"): "Long run high grey level emphasis",
    ("glrlm", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    (
        "glrlm",
        "GrayLevelNonUniformityNormalized",
    ): "Normalised grey level non-uniformity",
    ("glrlm", "RunLengthNonUniformity"): "Run length non-uniformity",
    (
        "glrlm",
        "RunLengthNonUniformityNormalized",
    ): "Normalised run length non-uniformity",
    ("glrlm", "RunPercentage"): "Run percentage",
    ("glrlm", "GrayLevelVariance"): "Grey level variance",
    ("glrlm", "RunVariance"): "Run length variance",
    ("glrlm", "RunEntropy"): "Run entropy",
    ("glszm", "SmallAreaEmphasis"): "Small zone emphasis",
    ("glszm", "LargeAreaEmphasis"): "Large zone emphasis",
    ("glszm", "LowGrayLevelZoneEmphasis"): "Low grey level zone emphasis",
    ("glszm", "HighGrayLevelZoneEmphasis"): "High grey level zone emphasis",
    ("glszm", "SmallAreaLowGrayLevelEmphasis"): "Small zone low grey level emphasis",
    ("glszm", "SmallAreaHighGrayLevelEmphasis"): "Small zone high grey level emphasis",
    ("glszm", "LargeAreaLowGrayLevelEmphasis"): "Large zone low grey level emphasis",
    ("glszm", "LargeAreaHighGrayLevelEmphasis"): "Large zone high grey level emphasis",
    ("glszm", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    (
        "glszm",
        "GrayLevelNonUniformityNormalized",
    ): "Normalised grey level non-uniformity",
    ("glszm", "SizeZoneNonUniformity"): "Zone size non-uniformity",
    ("glszm", "SizeZoneNonUniformityNormalized"): "Normalised zone size non-uniformity",
    ("glszm", "ZonePercentage"): "Zone percentage",
    ("glszm", "GrayLevelVariance"): "Grey level variance",
    ("glszm", "ZoneVariance"): "Zone size variance",
    ("glszm", "ZoneEntropy"): "Zone size entropy",
    ("gldm", "SmallDependenceEmphasis"): "Low dependence emphasis",
    ("gldm", "LargeDependenceEmphasis"): "High dependence emphasis",
    ("gldm", "LowGrayLevelEmphasis"): "Low grey level count emphasis",
    ("gldm", "HighGrayLevelEmphasis"): "High grey level count emphasis",
    (
        "gldm",
        "SmallDependenceLowGrayLevelEmphasis",
    ): "Low dependence low grey level emphasis",
    (
        "gldm",
        "SmallDependenceHighGrayLevelEmphasis",
    ): "Low dependence high grey level emphasis",
    (
        "gldm",
        "LargeDependenceLowGrayLevelEmphasis",
    ): "High dependence low grey level emphasis",
    (
        "gldm",
        "LargeDependenceHighGrayLevelEmphasis",
    ): "High dependence high grey level emphasis",
    ("gldm", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    ("gldm", "DependenceNonUniformity"): "Dependence count non-uniformity",
    (
        "gldm",
        "DependenceNonUniformityNormalized",
    ): "Normalised dependence count non-uniformity",
    ("gldm", "DependenceVariance"): "Dependence count variance",
    ("gldm", "DependenceEntropy"): "Dependence count entropy",
    ("gldm", "GrayLevelVariance"): "Grey level variance",
    ("ngtdm", "Coarseness"): "Coarseness",
    ("ngtdm", "Contrast"): "Contrast",
    ("ngtdm", "Busyness"): "Busyness",
    ("ngtdm", "Complexity"): "Complexity",
    ("ngtdm", "Strength"): "Strength",
}


# These are not wrapper-derived approximations.  PyRadiomics 3.1.0 explicitly
# documents each source output as mathematically identical to the second IBSI
# definition.  The deprecated names themselves intentionally return no value,
# so retain the active source token and disclose its shared semantic provenance.
# DependencePercentage is deliberately absent: PyRadiomics documents it as an
# invariant but emits no active scalar output for it.
PYRADIOMICS_DOCUMENTED_EXACT_ALIASES: Dict[str, Dict[str, str]] = {
    "original_glcm_DifferenceAverage": {
        "8S9J": "PyRadiomics documents Dissimilarity as mathematically equal to Difference Average",
    },
    "original_glcm_ClusterTendency": {
        "OEEB": "PyRadiomics documents Sum Variance as mathematically equal to Cluster Tendency",
    },
    "original_firstorder_Uniformity": {
        "5SPA": "PyRadiomics documents NGLDM normalised GLNU as mathematically equal to first-order Uniformity",
    },
}

PYRADIOMICS_EXACT_ALIAS_EVIDENCE = (
    "https://pyradiomics.readthedocs.io/en/v3.1.0/removedfeatures.html"
)

PYRADIOMICS_EXPLICIT_DEPRECATED_FEATURES = {
    ("shape", "Compactness1"),
    ("shape", "Compactness2"),
    ("shape", "SphericalDisproportion"),
}


PYRADIOMICS_FAMILY_TO_IBSI = {
    "shape": "morphology",
    "firstorder": None,
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldm": "ngldm",
    "ngtdm": "ngtdm",
}


PYRADIOMICS_CODE_TO_FEATURES: Dict[str, List[tuple[str, str]]] = {}
for (family, feat), ibsi_name in PYRADIOMICS_MAP.items():
    ibsi_family = PYRADIOMICS_FAMILY_TO_IBSI.get(family)
    code = _code_from_name_in_family(ibsi_name, ibsi_family)
    if code:
        PYRADIOMICS_CODE_TO_FEATURES.setdefault(code, []).append((family, feat))

for _source_name, _aliases in PYRADIOMICS_DOCUMENTED_EXACT_ALIASES.items():
    _source_parts = _source_name.split("_", 2)
    if len(_source_parts) != 3 or _source_parts[0] != "original":
        raise RuntimeError(f"Invalid PyRadiomics exact-alias source: {_source_name}")
    _source_feature = (_source_parts[1], _source_parts[2])
    for _alias_code in _aliases:
        PYRADIOMICS_CODE_TO_FEATURES.setdefault(_alias_code, []).append(_source_feature)


def map_pyradiomics(feature_name: str) -> Optional[str]:
    parts = feature_name.split("_")
    if len(parts) < 3:
        return None
    if parts[0] != "original":
        return None
    family = parts[1]
    feat = "_".join(parts[2:])
    ibsi_name = PYRADIOMICS_MAP.get((family, feat))
    if not ibsi_name:
        return None
    ibsi_family = PYRADIOMICS_FAMILY_TO_IBSI.get(family)
    return _code_from_name_in_family(ibsi_name, ibsi_family)


def pyradiomics_semantic_aliases(feature_name: str) -> Dict[str, str]:
    """Return documented exact IBSI identities for one emitted native token."""

    return dict(PYRADIOMICS_DOCUMENTED_EXACT_ALIASES.get(feature_name, {}))


def documented_semantic_aliases(adapter: str, feature_name: str) -> Dict[str, str]:
    """Return reviewed one-to-many semantic identities for an adapter output."""

    if str(adapter).strip().lower() == "pyradiomics":
        return pyradiomics_semantic_aliases(feature_name)
    return {}


def pyradiomics_feature_selection(codes: Iterable[str]) -> Dict[str, List[str]]:
    features: Dict[str, List[str]] = {}
    for code in codes:
        for family, feat in PYRADIOMICS_CODE_TO_FEATURES.get(code, []):
            features.setdefault(family, [])
            if feat not in features[family]:
                features[family].append(feat)
    for family in features:
        features[family].sort()
    return features


MIRP_PREFIX_MAP = {
    # morphology
    "morph_volume": "Volume",
    "morph_vol_approx": "Volume voxel counting",
    "morph_area_mesh": "Surface area",
    "morph_av": "Surface to volume ratio",
    "morph_comp_1": "Compactness 1",
    "morph_comp_2": "Compactness 2",
    "morph_sph_dispr": "Spherical disproportion",
    "morph_sphericity": "Sphericity",
    "morph_asphericity": "Asphericity",
    "morph_com": "Center of mass shift",
    "morph_integ_int": "Integrated intensity",
    "morph_diam": "Maximum 3D diameter",
    "morph_vol_dens_conv_hull": "Volume density (convex hull)",
    "morph_area_dens_conv_hull": "Area density (convex hull)",
    "morph_vol_dens_aabb": "Volume density (AABB)",
    "morph_area_dens_aabb": "Area density (AABB)",
    "morph_pca_maj_axis": "Major axis length",
    "morph_pca_min_axis": "Minor axis length",
    "morph_pca_least_axis": "Least axis length",
    "morph_pca_elongation": "Elongation",
    "morph_pca_flatness": "Flatness",
    "morph_vol_dens_aee": "Volume density (AEE)",
    "morph_area_dens_aee": "Area density (AEE)",
    "morph_vol_dens_ombb": "Volume density (OMBB)",
    "morph_area_dens_ombb": "Area density (OMBB)",
    "morph_vol_dens_mvee": "Volume density (MVEE)",
    "morph_area_dens_mvee": "Area density (MVEE)",
    "morph_moran_i": "Moran's I index",
    "morph_geary_c": "Geary's C measure",
    # local intensity
    "loc_peak_loc": "Local intensity peak ",
    "loc_peak_glob": "Global intensity peak",
    # intensity stats
    "stat_mean": "Mean intensity",
    "stat_var": "Intensity variance",
    "stat_skew": "Intensity skewness",
    "stat_kurt": "Intensity kurtosis",
    "stat_median": "Median intensity",
    "stat_min": "Minimum intensity",
    "stat_p10": "10th intensity percentile",
    "stat_p90": "90th intensity percentile",
    "stat_max": "Maximum intensity",
    "stat_iqr": "Intensity interquartile range",
    "stat_range": "Intensity range",
    "stat_mad": "Intensity Mean absolute deviation",
    "stat_rmad": "Intensity Robust mean absolute deviation",
    "stat_medad": "Intensity Median absolute deviation",
    "stat_cov": "Intensity Coefficient of variation",
    "stat_qcod": "Intensity Quartile coefficient of dispersion",
    "stat_energy": "Intensity energy",
    "stat_rms": "Root mean square intensity",
    # intensity histogram
    "ih_mean": "Mean discretised intensity",
    "ih_var": "Discretised intensity variance",
    "ih_skew": "Discretised intensity skewness",
    "ih_kurt": "Discretised intensity kurtosis",
    "ih_median": "Median discretised intensity",
    "ih_min": "Minimum discretised intensity",
    "ih_p10": "10th discretised intensity percentile",
    "ih_p90": "90th discretised intensity percentile",
    "ih_max": "Maximum discretised intensity",
    "ih_mode": "Intensity histogram mode",
    "ih_iqr": "Discretised intensity interquartile range",
    "ih_range": "Discretised intensity range",
    "ih_mad": "Intensity histogram mean absolute deviation",
    "ih_rmad": "Intensity histogram robust mean absolute deviation",
    "ih_medad": "Intensity histogram median absolute deviation",
    "ih_cov": "Intensity histogram coefficient of variation",
    "ih_qcod": "Intensity histogram quartile coefficient of dispersion",
    "ih_entropy": "Discretised intensity entropy",
    "ih_uniformity": "Discretised intensity uniformity",
    "ih_max_grad": "Maximum histogram gradient",
    "ih_max_grad_g": "Maximum histogram gradient intensity",
    "ih_min_grad": "Minimum histogram gradient",
    "ih_min_grad_g": "Minimum histogram gradient intensity",
    # IVH (only IBSI-defined percentiles map cleanly)
    "ivh_v10": "Volume at intensity fraction 0.10",
    "ivh_v90": "Volume at intensity fraction 0.90",
    "ivh_i10": "Intensity at volume fraction 0.10",
    "ivh_i90": "Intensity at volume fraction 0.90",
    "ivh_diff_v10_v90": "Volume fraction difference between intensity 0.10 and 0.90 fractions",
    "ivh_diff_i10_i90": "Intensity fraction difference between volume 0.10 and 0.90 fractions",
    "ivh_auc": "Area under the IVH curve",
    # GLCM (cm_*)
    "cm_joint_max": "Joint maximum",
    "cm_joint_avg": "Joint average",
    "cm_joint_var": "Joint variance",
    "cm_joint_entr": "Joint entropy",
    "cm_diff_avg": "Difference average",
    "cm_diff_var": "Difference variance",
    "cm_diff_entr": "Difference entropy",
    "cm_sum_avg": "Sum average",
    "cm_sum_var": "Sum variance",
    "cm_sum_entr": "Sum entropy",
    "cm_energy": "Angular second moment",
    "cm_contrast": "Contrast",
    "cm_dissimilarity": "Dissimilarity",
    "cm_inv_diff": "Inverse difference",
    "cm_inv_diff_norm": "Normalised inverse difference",
    "cm_inv_diff_mom": "Inverse difference moment",
    "cm_inv_diff_mom_norm": "Normalised inverse difference moment",
    "cm_inv_var": "Inverse variance",
    "cm_corr": "Correlation",
    "cm_auto_corr": "Autocorrelation",
    "cm_clust_tend": "Cluster tendency",
    "cm_clust_shade": "Cluster shade",
    "cm_clust_prom": "Cluster prominence",
    "cm_info_corr1": "Information correlation 1",
    "cm_info_corr2": "Information correlation 2",
    # GLRLM (rlm_*)
    "rlm_sre": "Short runs emphasis",
    "rlm_lre": "Long runs emphasis",
    "rlm_lgre": "Low grey level run emphasis",
    "rlm_hgre": "High grey level run emphasis",
    "rlm_srlge": "Short run low grey level emphasis",
    "rlm_srhge": "Short run high grey level emphasis",
    "rlm_lrlge": "Long run low grey level emphasis",
    "rlm_lrhge": "Long run high grey level emphasis",
    "rlm_glnu": "Grey level non-uniformity",
    "rlm_glnu_norm": "Normalised grey level non-uniformity",
    "rlm_rlnu": "Run length non-uniformity",
    "rlm_rlnu_norm": "Normalised run length non-uniformity",
    "rlm_r_perc": "Run percentage",
    "rlm_gl_var": "Grey level variance",
    "rlm_rl_var": "Run length variance",
    "rlm_rl_entr": "Run entropy",
    # GLSZM (szm_*)
    "szm_sze": "Small zone emphasis",
    "szm_lze": "Large zone emphasis",
    "szm_lgze": "Low grey level zone emphasis",
    "szm_hgze": "High grey level zone emphasis",
    "szm_szlge": "Small zone low grey level emphasis",
    "szm_szhge": "Small zone high grey level emphasis",
    "szm_lzlge": "Large zone low grey level emphasis",
    "szm_lzhge": "Large zone high grey level emphasis",
    "szm_glnu": "Grey level non-uniformity",
    "szm_glnu_norm": "Normalised grey level non-uniformity",
    "szm_zsnu": "Zone size non-uniformity",
    "szm_zsnu_norm": "Normalised zone size non-uniformity",
    "szm_z_perc": "Zone percentage",
    "szm_gl_var": "Grey level variance",
    "szm_zs_var": "Zone size variance",
    "szm_zs_entr": "Zone size entropy",
    # GLDZM (dzm_*)
    "dzm_sde": "Small distance emphasis",
    "dzm_lde": "Large distance emphasis",
    "dzm_lgze": "Low grey level zone emphasis",
    "dzm_hgze": "High grey level zone emphasis",
    "dzm_sdlge": "Small distance low grey level emphasis",
    "dzm_sdhge": "Small distance high grey level emphasis",
    "dzm_ldlge": "Large distance low grey level emphasis",
    "dzm_ldhge": "Large distance high grey level emphasis",
    "dzm_glnu": "Grey level non-uniformity",
    "dzm_glnu_norm": "Normalised grey level non-uniformity",
    "dzm_zdnu": "Zone distance non-uniformity",
    "dzm_zdnu_norm": "Normalised zone distance non-uniformity",
    "dzm_z_perc": "Zone percentage",
    "dzm_gl_var": "Grey level variance",
    "dzm_zd_var": "Zone distance variance",
    "dzm_zd_entr": "Zone distance entropy",
    # NGTDM (ngt_*)
    "ngt_coarseness": "Coarseness",
    "ngt_contrast": "Contrast",
    "ngt_busyness": "Busyness",
    "ngt_complexity": "Complexity",
    "ngt_strength": "Strength",
    # NGLDM (ngl_*)
    "ngl_lde": "Low dependence emphasis",
    "ngl_hde": "High dependence emphasis",
    "ngl_lgce": "Low grey level count emphasis",
    "ngl_hgce": "High grey level count emphasis",
    "ngl_ldlge": "Low dependence low grey level emphasis",
    "ngl_ldhge": "Low dependence high grey level emphasis",
    "ngl_hdlge": "High dependence low grey level emphasis",
    "ngl_hdhge": "High dependence high grey level emphasis",
    "ngl_glnu": "Grey level non-uniformity",
    "ngl_glnu_norm": "Normalised grey level non-uniformity",
    "ngl_dcnu": "Dependence count non-uniformity",
    "ngl_dcnu_norm": "Normalised dependence count non-uniformity",
    "ngl_dc_perc": "Dependence count percentage",
    "ngl_gl_var": "Grey level variance",
    "ngl_dc_var": "Dependence count variance",
    "ngl_dc_entr": "Dependence count entropy",
    "ngl_dc_energy": "Dependence count energy",
}


MIRP_PREFIX_TO_FAMILY = {
    "morph_": "morphology",
    "loc_": "local_intensity",
    "stat_": "intensity",
    "ih_": "histogram",
    "ivh_": "ivh",
    "cm_": "glcm",
    "rlm_": "glrlm",
    "szm_": "glszm",
    "dzm_": "gldzm",
    "ngt_": "ngtdm",
    "ngl_": "ngldm",
}


def _mirp_family_for_prefix(prefix: str) -> Optional[str]:
    for key, family in MIRP_PREFIX_TO_FAMILY.items():
        if prefix.startswith(key):
            return family
    return None


def map_mirp(feature_name: str) -> Optional[str]:
    # Remove discretization suffixes like _fbn_n32
    base = feature_name
    base = re.sub(r"_fbn_n\d+", "", base)
    base = re.sub(r"_fbs_w[0-9.]+", "", base)
    # MIRP's offset energy and RMS variants are not IBSI features.
    if base.endswith("_offset"):
        return None

    # MIRP feature names can include extra tokens; find longest prefix match
    for prefix in sorted(MIRP_PREFIX_MAP, key=len, reverse=True):
        ibsi_name = MIRP_PREFIX_MAP[prefix]
        if base.startswith(prefix):
            family = _mirp_family_for_prefix(prefix)
            return _code_from_name_in_family(ibsi_name, family)

    return None


RADIOMICS_DEVELOP_TOKEN_ALIASES: Dict[str, str] = {
    # Morphology aliases relative to MIRP naming.
    "morph_vol": "morph_volume",
    "morph_approx_vol": "morph_vol_approx",
    "morph_area": "morph_area_mesh",
    "morph_pca_major": "morph_pca_maj_axis",
    "morph_pca_minor": "morph_pca_min_axis",
    "morph_pca_least": "morph_pca_least_axis",
    "morph_v_dens_aabb": "morph_vol_dens_aabb",
    "morph_v_dens_aee": "morph_vol_dens_aee",
    "morph_v_dens_conv_hull": "morph_vol_dens_conv_hull",
    "morph_a_dens_aabb": "morph_area_dens_aabb",
    "morph_a_dens_aee": "morph_area_dens_aee",
    "morph_a_dens_conv_hull": "morph_area_dens_conv_hull",
    # Local intensity aliases.
    "loc_peak_local": "loc_peak_loc",
    "loc_peak_global": "loc_peak_glob",
    # Median absolute deviation naming differs across snapshots.
    "stat_medmad": "stat_medad",
    "ih_medmad": "ih_medad",
    # Histogram gradient suffix aliases.
    "ih_max_grad_gl": "ih_max_grad_g",
    "ih_min_grad_gl": "ih_min_grad_g",
    # IVH difference aliases.
    "ivh_v10minusv90": "ivh_diff_v10_v90",
    "ivh_i10minusi90": "ivh_diff_i10_i90",
    # GLCM information correlation aliases.
    "cm_info_corr_1": "cm_info_corr1",
    "cm_info_corr_2": "cm_info_corr2",
    "cm_info_corr1": "cm_info_corr1",
    "cm_info_corr2": "cm_info_corr2",
}

RADIOMICS_DEVELOP_DIRECT_MAP: Dict[str, tuple[str, str]] = {
    # Present in radiomics-develop outputs but not MIRP prefix map.
    "morph_v_dens_ombb": ("Volume density (OMBB)", "morphology"),
    "morph_a_dens_ombb": ("Area density (OMBB)", "morphology"),
    "morph_v_dens_mvee": ("Volume density (MVEE)", "morphology"),
    "morph_a_dens_mvee": ("Area density (MVEE)", "morphology"),
}


def map_radiomics_develop(feature_name: str) -> Optional[str]:
    token = feature_name.strip()
    if token.startswith("F"):
        token = token[1:]
    token = token.lower()

    direct = RADIOMICS_DEVELOP_DIRECT_MAP.get(token)
    if direct:
        ibsi_name, family = direct
        return _code_from_name_in_family(ibsi_name, family)

    alias = RADIOMICS_DEVELOP_TOKEN_ALIASES.get(token, token)
    code = map_mirp(alias)
    if code:
        return code

    # Fallback for sparse naming variants.
    alias = alias.replace("__", "_")
    return map_mirp(alias)


def map_medimage(feature_name: str) -> Optional[str]:
    # MEDimage output naming is MIRP/radiomics-develop aligned.
    return map_radiomics_develop(feature_name)


def map_zrad(feature_name: str) -> Optional[str]:
    token = feature_name.strip().lower()
    if token.startswith("f") and "_" in token:
        token = token[1:]

    # Aggregation suffixes used by Z-Rad texture outputs.
    for suffix in (
        "_3d_comb",
        "_3d_avg",
        "_3d",
        "_2_5d_comb",
        "_2_5d_avg",
        "_2_5d",
        "_2d_comb",
        "_2d_avg",
        "_2d",
    ):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break

    # Z-Rad feature tokens are MIRP/radiomics-develop aligned.
    return map_mirp(token)


PYRADIOMICS_EXCLUDE = {
    "original_firstorder_TotalEnergy",  # not in IBSI glossary
    "original_glcm_MCC",  # maximal correlation coefficient not in IBSI
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice",
}

MIRP_IVH_EXCLUDE = {
    "ivh_v25",
    "ivh_v50",
    "ivh_v75",
    "ivh_i25",
    "ivh_i50",
    "ivh_i75",
    "ivh_diff_v25_v75",
    "ivh_diff_i25_i75",
}


def _mirp_excluded(feature_name: str) -> bool:
    base = re.sub(r"_fbn_n\d+", "", feature_name)
    base = re.sub(r"_fbs_w[0-9.]+", "", base)
    if base.startswith("image_") or base == "sample_name":
        return True
    if base in MIRP_IVH_EXCLUDE:
        return True
    return False


ZRAD_EXCLUDE = {
    "bounding_box_min",
    "no_voxels",
    "no_bins",
}


def _zrad_excluded(feature_name: str) -> bool:
    token = feature_name.strip().lower()
    if token.startswith("f") and "_" in token:
        token = token[1:]
    if token in ZRAD_EXCLUDE:
        return True
    return False


def classify_feature(adapter: str, feature_name: str) -> tuple[Optional[str], str]:
    """
    Returns (ibsi_code, status) where status is: mapped, excluded, or unmapped.
    """
    if adapter == "pictologics":
        code = map_pictologics(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "pyradiomics":
        if feature_name in PYRADIOMICS_EXCLUDE:
            return (None, "excluded")
        code = map_pyradiomics(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "mirp":
        if _mirp_excluded(feature_name):
            return (None, "excluded")
        code = map_mirp(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "medimage":
        code = map_medimage(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "zrad":
        if _zrad_excluded(feature_name):
            return (None, "excluded")
        code = map_zrad(feature_name)
        return (code, "mapped" if code else "unmapped")

    return (None, "unmapped")


IBSI_FAMILY_TO_PICTOLOGICS = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_MIRP = {
    "morphology": "morphology",
    "morphology_shape": "morphology",
    "morphology_spatial": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "statistics",
    "histogram": "intensity_histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}


IBSI_FAMILY_TO_MEDIMAGE = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_ZRAD = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}


def pictologics_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_PICTOLOGICS.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def mirp_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_MIRP.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def medimage_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_MEDIMAGE.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def zrad_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_ZRAD.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)
