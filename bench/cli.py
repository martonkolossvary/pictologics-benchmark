from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench import env


DEFAULT_ADAPTERS = "pictologics,pyradiomics,mirp,medimage,zrad"


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset folder that contains manifest.json",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--report-dir", default=None, help="Run-state/output directory")
    parser.add_argument("--adapters", default=DEFAULT_ADAPTERS)
    parser.add_argument(
        "--sizes", default=None, help="Optional comma-separated synthetic size filter"
    )
    parser.add_argument(
        "--variants", default=None, help="Optional comma-separated variant filter"
    )
    parser.add_argument(
        "--masks", default=None, help="Optional comma-separated mask-ID filter"
    )
    parser.add_argument(
        "--modalities",
        default=None,
        help="Optional comma-separated modality filter",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Fresh adapter processes per eligible case/family (reviewed default: 3)",
    )
    parser.add_argument(
        "--endpoint-contract",
        default="configs/benchmark/main.json",
    )
    parser.add_argument(
        "--input-contract",
        choices=["auto", "manifest_harmonized", "adapter_internal_preprocessing"],
        default="auto",
    )
    parser.add_argument("--discretization", choices=["fbn", "fbs"], default="fbn")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--bin-width", type=float, default=32.0)
    parser.add_argument("--intensity-min", type=float, default=None)
    parser.add_argument("--intensity-max", type=float, default=None)
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated canonical families scheduled independently, or 'all'",
    )
    parser.add_argument("--timing-iterations", type=int, default=3)
    parser.add_argument("--drop-first-repeat", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--termination-grace", type=float, default=3.0)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-repeats", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--project-total-tasks", type=int, default=None)
    parser.add_argument("--project-task-offset", type=int, default=0)
    parser.add_argument(
        "--machine-id",
        default=None,
        help="Stable, non-identifying machine ID for multi-host studies",
    )
    parser.add_argument(
        "--machine-label",
        default=None,
        help="Optional reader-facing machine label",
    )
    parser.add_argument(
        "--cpu-model",
        default=None,
        help="Override an unavailable or generic automatic CPU model",
    )
    parser.add_argument(
        "--cpu-base-ghz",
        type=float,
        default=None,
        help="Optional documented CPU base frequency in GHz",
    )
    parser.add_argument("--host-profile-id", default=None)
    parser.add_argument("--host-profile-sha256", default=None)
    parser.add_argument("--host-settings-json", default=None)
    parser.add_argument("--guardrail-baseline", default="pictologics")
    parser.add_argument("--guardrail-skip-ratio", type=float, default=1000.0)
    parser.add_argument("--guardrail-min-observations", type=int, default=1)
    parser.add_argument("--disable-speed-truncation", action="store_true")
    parser.add_argument("--no-truncate-on-timeout", action="store_true")
    parser.add_argument("--disable-memory-preflight", action="store_true")
    parser.add_argument("--memory-budget-fraction", type=float, default=0.80)
    parser.add_argument("--memory-reserve-gib", type=float, default=4.0)
    parser.add_argument("--memory-cap-gib", type=float, default=None)
    parser.add_argument("--memory-safety-factor", type=float, default=1.50)
    parser.add_argument("--no-verify-dataset-hashes", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the immutable plan only",
    )


def _run_argv(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--dataset-dir",
        args.dataset_dir,
        "--repeats",
        str(args.repeats),
        "--endpoint-contract",
        args.endpoint_contract,
        "--input-contract",
        args.input_contract,
        "--discretization",
        args.discretization,
        "--bins",
        str(args.bins),
        "--bin-width",
        str(args.bin_width),
        "--adapters",
        args.adapters,
        "--timing-iterations",
        str(args.timing_iterations),
        "--termination-grace",
        str(args.termination_grace),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--progress-interval",
        str(args.progress_interval),
        "--project-task-offset",
        str(args.project_task_offset),
        "--guardrail-baseline",
        args.guardrail_baseline,
        "--guardrail-skip-ratio",
        str(args.guardrail_skip_ratio),
        "--guardrail-min-observations",
        str(args.guardrail_min_observations),
        "--memory-budget-fraction",
        str(args.memory_budget_fraction),
        "--memory-reserve-gib",
        str(args.memory_reserve_gib),
        "--memory-safety-factor",
        str(args.memory_safety_factor),
    ]
    optional_values = (
        ("--run-id", args.run_id),
        ("--report-dir", args.report_dir),
        ("--sizes", args.sizes),
        ("--variants", args.variants),
        ("--masks", args.masks),
        ("--modalities", args.modalities),
        ("--families", args.families),
        ("--timeout", args.timeout),
        ("--intensity-min", args.intensity_min),
        ("--intensity-max", args.intensity_max),
        ("--memory-cap-gib", args.memory_cap_gib),
        ("--project-total-tasks", args.project_total_tasks),
        ("--machine-id", args.machine_id),
        ("--machine-label", args.machine_label),
        ("--cpu-model", args.cpu_model),
        ("--cpu-base-ghz", args.cpu_base_ghz),
        ("--host-profile-id", args.host_profile_id),
        ("--host-profile-sha256", args.host_profile_sha256),
        ("--host-settings-json", args.host_settings_json),
    )
    for flag, value in optional_values:
        if value is not None:
            forwarded.extend([flag, str(value)])
    boolean_flags = (
        ("--drop-first-repeat", args.drop_first_repeat),
        ("--no-warmup", args.no_warmup),
        ("--keep-going", args.keep_going),
        ("--resume", args.resume),
        ("--extend-repeats", args.extend_repeats),
        ("--disable-speed-truncation", args.disable_speed_truncation),
        ("--no-truncate-on-timeout", args.no_truncate_on_timeout),
        ("--disable-memory-preflight", args.disable_memory_preflight),
        ("--no-verify-dataset-hashes", args.no_verify_dataset_hashes),
        ("--dry-run", args.dry_run),
    )
    for flag, enabled in boolean_flags:
        if enabled:
            forwarded.append(flag)
    return forwarded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Version-isolated, resumable Python radiomics benchmarks",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pillar1 = commands.add_parser(
        "generate-pillar1",
        help="Build/resume the frozen public Pillar 1 raw + FBN32 + IVH dataset",
    )
    pillar1.add_argument("--dest-dir", default="data/benchmark/pillar1")
    pillar1.add_argument("--profile", default="configs/benchmark/pillar1.json")
    pillar1.add_argument("--resume", action="store_true")
    pillar1.add_argument("--validate-only", action="store_true")
    pillar1.add_argument("--shallow-validation", action="store_true")

    pillar2 = commands.add_parser(
        "generate-pillar2",
        help="Build the dense whole-anatomy A1 scaling dataset",
    )
    pillar2.add_argument("--dest-dir", default="data/benchmark/pillar2_a1")
    pillar2.add_argument("--profile", default="configs/benchmark/pillar2.json")
    pillar2.add_argument("--resume", action="store_true")
    pillar2.add_argument("--validate-only", action="store_true")

    phase3 = commands.add_parser(
        "prepare-ibsi2-phase3",
        help="Copy and attest the fixed IBSI 2 Phase 3 cohort inside the workspace",
    )
    phase3.add_argument("--source-dir", required=True)
    phase3.add_argument("--dest-dir", default="data/benchmark/ibsi2_phase3")
    phase3.add_argument("--expected-subjects", type=int, default=51)
    phase3.add_argument("--resume", action="store_true")
    phase3.add_argument("--validate-only", action="store_true")

    validate = commands.add_parser(
        "validate-dataset", help="Verify a dataset manifest, files, and geometry"
    )
    validate.add_argument("--dataset-dir", required=True)
    validate.add_argument("--no-verify-hashes", action="store_true")
    validate.add_argument("--no-inspect-geometry", action="store_true")

    environment = commands.add_parser(
        "env", help="Manage isolated adapter environments"
    )
    environment_commands = environment.add_subparsers(dest="env_command", required=True)
    environment_commands.add_parser("list")
    create = environment_commands.add_parser("create")
    create.add_argument("--profiles", nargs="*")
    create.add_argument("--force", action="store_true")
    verify = environment_commands.add_parser("verify")
    verify.add_argument("--profiles", nargs="*")

    benchmark = commands.add_parser(
        "run", help="Execute or resume a benchmark task plan"
    )
    _add_run_arguments(benchmark)

    render = commands.add_parser("report", help="Render reports from a benchmark run")
    render.add_argument("--input-dir", required=True)
    render.add_argument("--output-dir", default=None)

    compliance = commands.add_parser(
        "compliance",
        help="Import, run, resume, evaluate, and report IBSI 1/2 compliance",
    )
    compliance_commands = compliance.add_subparsers(
        dest="compliance_command", required=True
    )

    ibsi1_import = compliance_commands.add_parser(
        "import-ibsi1", help="Strictly import the official six-sheet IBSI 1 workbook"
    )
    ibsi1_import.add_argument("--workbook", required=True)
    ibsi1_import.add_argument("--output-dir", required=True)
    ibsi1_import.add_argument("--allow-unknown-hash", action="store_true")

    ibsi2_import = compliance_commands.add_parser(
        "import-ibsi2-phase2", help="Import the reviewed analysis-derived Phase 2 table"
    )
    ibsi2_import.add_argument("--csv", required=True)
    ibsi2_import.add_argument("--output-dir", required=True)
    ibsi2_import.add_argument("--allow-unknown-hash", action="store_true")

    phase1_validate = compliance_commands.add_parser(
        "validate-ibsi2-phase1",
        help="Validate the exact 33-map standardized reference bundle",
    )
    phase1_validate.add_argument("--reference-dir", required=True)
    phase1_validate.add_argument("--manifest-out", required=True)

    ibsi1_run = compliance_commands.add_parser(
        "run-ibsi1",
        help=(
            "Run/resume the 172-definition/174-instance/169-standardized "
            "digital phantom suite using 3D-merged aggregation"
        ),
    )
    ibsi1_run.add_argument("--image", required=True)
    ibsi1_run.add_argument("--mask", required=True)
    ibsi1_run.add_argument("--references", required=True)
    ibsi1_run.add_argument("--reference-manifest", required=True)
    ibsi1_run.add_argument("--output-dir", required=True)
    ibsi1_run.add_argument("--adapters", default=DEFAULT_ADAPTERS)
    ibsi1_run.add_argument("--timeout", type=float, default=None)
    ibsi1_run.add_argument("--resume", action="store_true")
    ibsi1_run.add_argument("--no-report", action="store_true")

    ibsi2_generate = compliance_commands.add_parser(
        "generate-ibsi2-candidates",
        help=(
            "Generate/resume package-native IBSI 2 response maps from the pinned "
            "official phantoms"
        ),
    )
    ibsi2_generate.add_argument("--output-dir", required=True)
    ibsi2_generate.add_argument("--adapters", default=DEFAULT_ADAPTERS)
    ibsi2_generate.add_argument("--phases", default="phase1,phase2")
    ibsi2_generate.add_argument("--timeout", type=float, default=1800.0)
    ibsi2_generate.add_argument("--resume", action="store_true")

    phase1_evaluate = compliance_commands.add_parser(
        "evaluate-ibsi2-phase1",
        help="Evaluate package response maps against a validated Phase 1 reference manifest",
    )
    phase1_evaluate.add_argument("--reference-manifest", required=True)
    phase1_evaluate.add_argument("--reference-dir", required=True)
    phase1_evaluate.add_argument(
        "--candidate-manifest",
        required=True,
        help="Checksummed package/version/config manifest of candidate response maps",
    )
    phase1_evaluate.add_argument("--adapters", default=DEFAULT_ADAPTERS)
    phase1_evaluate.add_argument("--output-dir", required=True)

    phase2_run = compliance_commands.add_parser(
        "run-ibsi2-phase2",
        help="Run/resume statistics on provenance-attested package response maps",
    )
    phase2_run.add_argument("--candidate-manifest", required=True)
    phase2_run.add_argument("--references", required=True)
    phase2_run.add_argument("--reference-manifest", required=True)
    phase2_run.add_argument("--output-dir", required=True)
    phase2_run.add_argument("--adapters", default=DEFAULT_ADAPTERS)
    phase2_run.add_argument("--timeout", type=float, default=None)
    phase2_run.add_argument("--resume", action="store_true")
    phase2_run.add_argument("--no-report", action="store_true")

    compliance_report = compliance_commands.add_parser(
        "report", help="Regenerate accessible tables/figures from comparison rows"
    )
    compliance_report.add_argument("--comparisons", required=True)
    compliance_report.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "generate-pillar1":
        from bench.pillar1_dataset import (
            build_pillar1_dataset,
            validate_pillar1_dataset,
        )

        destination = Path(args.dest_dir)
        if not args.validate_only:
            build_pillar1_dataset(
                destination,
                profile_path=Path(args.profile),
                resume=args.resume,
            )
        summary = validate_pillar1_dataset(
            destination,
            deep=not args.shallow_validation,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "generate-pillar2":
        from bench.pillar2_dataset import (
            build_pillar2_dataset,
            validate_pillar2_dataset,
        )

        destination = Path(args.dest_dir)
        if not args.validate_only:
            build_pillar2_dataset(
                destination,
                profile_path=Path(args.profile),
                resume=args.resume,
            )
        summary = validate_pillar2_dataset(destination)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-ibsi2-phase3":
        from bench.ibsi2_phase3_dataset import (
            prepare_ibsi2_phase3_dataset,
            validate_ibsi2_phase3_dataset,
        )

        destination = Path(args.dest_dir)
        if not args.validate_only:
            prepare_ibsi2_phase3_dataset(
                Path(args.source_dir),
                destination,
                expected_subjects=args.expected_subjects,
                resume=args.resume,
            )
        summary = validate_ibsi2_phase3_dataset(destination)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-dataset":
        from bench.dataset_manifest import load_and_validate_manifest

        _, summary = load_and_validate_manifest(
            Path(args.dataset_dir),
            verify_hashes=not args.no_verify_hashes,
            inspect_geometry=not args.no_inspect_geometry,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "env":
        forwarded = [args.env_command]
        if args.env_command in {"create", "verify"} and args.profiles:
            forwarded.extend(["--profiles", *args.profiles])
        if args.env_command == "create" and args.force:
            forwarded.append("--force")
        return env.main(forwarded)

    if args.command == "run":
        from bench import run

        return run.main(_run_argv(args))

    if args.command == "report":
        from bench import report

        forwarded = ["--input-dir", args.input_dir]
        if args.output_dir:
            forwarded.extend(["--output-dir", args.output_dir])
        return report.main(forwarded)

    if args.command == "compliance":
        from bench.compliance import references as compliance_references

        if args.compliance_command == "import-ibsi1":
            manifest = compliance_references.import_ibsi1_workbook(
                Path(args.workbook),
                Path(args.output_dir),
                require_known_hash=not args.allow_unknown_hash,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        if args.compliance_command == "import-ibsi2-phase2":
            manifest = compliance_references.import_ibsi2_phase2_csv(
                Path(args.csv),
                Path(args.output_dir),
                require_reviewed_derived_hash=not args.allow_unknown_hash,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        if args.compliance_command == "validate-ibsi2-phase1":
            manifest = compliance_references.validate_ibsi2_phase1_bundle(
                Path(args.reference_dir),
                manifest_path=Path(args.manifest_out),
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        if args.compliance_command == "run-ibsi1":
            from bench.compliance.run import run_ibsi1_digital_phantom

            if args.timeout is not None and args.timeout <= 0:
                raise ValueError("--timeout must be positive")
            adapters = [
                value.strip() for value in args.adapters.split(",") if value.strip()
            ]
            records = run_ibsi1_digital_phantom(
                image=Path(args.image),
                mask=Path(args.mask),
                references_csv=Path(args.references),
                reference_manifest=Path(args.reference_manifest),
                output_dir=Path(args.output_dir),
                adapters=adapters,
                resume=args.resume,
                timeout=args.timeout,
                render_report=not args.no_report,
            )
            print(json.dumps({"comparison_rows": len(records)}, indent=2))
            return 0

        if args.compliance_command == "generate-ibsi2-candidates":
            from bench.compliance.ibsi2_candidates import generate_candidate_bundle

            if args.timeout is not None and args.timeout <= 0:
                raise ValueError("--timeout must be positive")
            adapters = [
                value.strip() for value in args.adapters.split(",") if value.strip()
            ]
            phases = [
                value.strip() for value in args.phases.split(",") if value.strip()
            ]
            result = generate_candidate_bundle(
                output_dir=Path(args.output_dir),
                adapters=adapters,
                phases=phases,
                resume=args.resume,
                timeout=args.timeout,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.compliance_command == "evaluate-ibsi2-phase1":
            from bench.benchmark_ledger import atomic_write_json, sha256_file
            from bench.compliance.evaluate import evaluate_ibsi2_phase1_candidates
            from bench.compliance.report import generate_ibsi2_phase1_report
            from bench.compliance.run import (
                _require_requested_adapters,
                configured_adapter_profiles,
                ibsi2_execution_contracts,
                load_ibsi2_phase1_candidate_manifest,
            )

            with Path(args.reference_manifest).open("r", encoding="utf-8") as stream:
                reference_manifest = json.load(stream)
            adapters = [
                value.strip() for value in args.adapters.split(",") if value.strip()
            ]
            if not adapters:
                raise ValueError("At least one adapter is required")
            adapter_profiles = configured_adapter_profiles(adapters)
            candidate_entries = load_ibsi2_phase1_candidate_manifest(
                Path(args.candidate_manifest)
            )
            _require_requested_adapters(
                adapters, candidate_entries.adapters, phase="Phase 1"
            )
            support_declarations = candidate_entries.support_declarations
            execution_contracts = ibsi2_execution_contracts(
                candidate_entries,
                id_field="test_id",
            )
            output = Path(args.output_dir).expanduser().resolve()
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    "IBSI 2 Phase 1 output directory is not empty; use a new directory"
                )
            rows = []
            for adapter in adapters:
                candidate_paths = {
                    entry["test_id"]: [entry["response_map"]]
                    for entry in candidate_entries
                    if entry["adapter"] == adapter
                }
                candidate_metadata = {
                    entry["test_id"]: entry
                    for entry in candidate_entries
                    if entry["adapter"] == adapter
                }
                adapter_rows = evaluate_ibsi2_phase1_candidates(
                    adapter=adapter,
                    reference_manifest=reference_manifest,
                    reference_root=Path(args.reference_dir),
                    candidate_paths=candidate_paths,
                    candidate_metadata=candidate_metadata,
                )
                for row in adapter_rows:
                    declaration = support_declarations[(adapter, row["test_id"])]
                    row["support_class"] = candidate_entries.support_class
                    row["native_supported"] = declaration["native_supported"]
                    row["native_support_reason"] = declaration["reason"]
                    row["native_support_evidence"] = declaration["evidence"]
                    if not declaration["native_supported"]:
                        row.update(
                            status="native_unsupported",
                            detail=(
                                f"reviewed native-support declaration: {declaration['reason']}; "
                                f"evidence: {declaration['evidence']}"
                            ),
                            candidate_supplied=False,
                            supported=False,
                        )
                rows.extend(adapter_rows)
            missing_supported_candidates = [
                {"adapter": row["adapter"], "test_id": row["test_id"]}
                for row in rows
                if row.get("native_supported") and not row.get("candidate_supplied")
            ]
            expected_evaluated = 33 * len(adapters)
            processing_failures = [
                {
                    "adapter": row["adapter"],
                    "test_id": row["test_id"],
                    "status": row["status"],
                }
                for row in rows
                if row.get("native_supported")
                and row.get("standardized")
                and not row.get("evaluated")
            ]
            publication_complete = (
                not missing_supported_candidates and not processing_failures
            )
            output.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output / "phase1_results.json", rows)
            atomic_write_json(
                output / "result_manifest.json",
                {
                    "schema_version": 1,
                    "kind": "ibsi2_phase1_response_map_compliance",
                    "support_class": candidate_entries.support_class,
                    "reference_manifest_name": Path(args.reference_manifest).name,
                    "reference_manifest_sha256": sha256_file(
                        Path(args.reference_manifest)
                    ),
                    "candidate_manifest_name": Path(args.candidate_manifest).name,
                    "candidate_manifest_sha256": sha256_file(
                        Path(args.candidate_manifest)
                    ),
                    "adapters": adapters,
                    "configured_adapter_profiles": adapter_profiles,
                    "execution_contracts": execution_contracts,
                    "defined_rows": len(rows),
                    "evaluated_rows": sum(bool(row.get("evaluated")) for row in rows),
                    "passed_rows": sum(row.get("passed") is True for row in rows),
                    "standardized_rows": expected_evaluated,
                    "support_declaration_grid_complete": True,
                    "native_supported_filter_tests": {
                        adapter: sum(
                            support_declarations[(adapter, test_id)]["native_supported"]
                            for test_id in compliance_references.IBSI2_PHASE1_TEST_IDS
                        )
                        for adapter in adapters
                    },
                    "native_filter_denominator": len(
                        compliance_references.IBSI2_PHASE1_TEST_IDS
                    ),
                    "missing_supported_candidate_maps": missing_supported_candidates,
                    "processing_failures": processing_failures,
                    "all_native_supported_candidate_maps_supplied": not missing_supported_candidates,
                    "publication_complete": publication_complete,
                    "results_sha256": sha256_file(output / "phase1_results.json"),
                },
            )
            manifest = generate_ibsi2_phase1_report(
                rows,
                output / "report",
                source_metadata={
                    "reference_manifest_name": Path(args.reference_manifest).name,
                    "reference_manifest_sha256": sha256_file(
                        Path(args.reference_manifest)
                    ),
                    "candidate_manifest_name": Path(args.candidate_manifest).name,
                    "candidate_manifest_sha256": sha256_file(
                        Path(args.candidate_manifest)
                    ),
                    "configured_adapter_profiles": adapter_profiles,
                    "execution_contracts": execution_contracts,
                    "support_declaration_grid_complete": True,
                    "native_supported_filter_tests": {
                        adapter: sum(
                            support_declarations[(adapter, test_id)]["native_supported"]
                            for test_id in compliance_references.IBSI2_PHASE1_TEST_IDS
                        )
                        for adapter in adapters
                    },
                    "native_filter_denominator": len(
                        compliance_references.IBSI2_PHASE1_TEST_IDS
                    ),
                    "missing_supported_candidate_maps": missing_supported_candidates,
                    "processing_failures": processing_failures,
                    "publication_complete": publication_complete,
                },
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        if args.compliance_command == "run-ibsi2-phase2":
            from bench.compliance.run import run_ibsi2_phase2_from_response_maps

            if args.timeout is not None and args.timeout <= 0:
                raise ValueError("--timeout must be positive")
            adapters = [
                value.strip() for value in args.adapters.split(",") if value.strip()
            ]
            records = run_ibsi2_phase2_from_response_maps(
                candidate_manifest=Path(args.candidate_manifest),
                references_csv=Path(args.references),
                reference_manifest=Path(args.reference_manifest),
                output_dir=Path(args.output_dir),
                adapters=adapters,
                resume=args.resume,
                timeout=args.timeout,
                render_report=not args.no_report,
            )
            print(json.dumps({"comparison_rows": len(records)}, indent=2))
            return 0

        if args.compliance_command == "report":
            from bench.compliance.report import (
                generate_compliance_report,
                load_comparison_csv,
            )

            records = load_comparison_csv(Path(args.comparisons))
            manifest = generate_compliance_report(records, Path(args.output_dir))
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
