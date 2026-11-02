from __future__ import annotations

import copy
import hashlib
import unittest

from bench import env


class EnvironmentProfileTests(unittest.TestCase):
    def test_latest_stable_adapter_pins_are_exact(self) -> None:
        profiles = env.load_runtime_profiles()
        self.assertEqual(
            {name: profile.version for name, profile in profiles.items()},
            {
                "medimage": "0.9.8",
                "mirp": "2.7.0",
                "pictologics": "1.0.0",
                "pyradiomics": "3.1.0",
                "zrad": "26.8.0",
            },
        )
        for profile in profiles.values():
            self.assertTrue(profile.verified_latest_stable)
            self.assertTrue(profile.upstream.startswith("https://"))

    def test_every_environment_is_isolated_below_adapter_root(self) -> None:
        base = (env.repo_root() / ".venvs" / "adapters").resolve()
        paths = []
        for profile in env.load_runtime_profiles().values():
            target = env.env_dir_for_profile(profile)
            target.relative_to(base)
            paths.append(target)
        self.assertEqual(len(paths), len(set(paths)))

    def test_escaping_environment_path_is_rejected(self) -> None:
        profile = env.RuntimeProfile(
            name="bad",
            distribution="bad",
            version="1",
            python="3.12",
            requirement="bad==1",
            env_dir="../outside",
            smoke_imports=(),
        )
        with self.assertRaises(env.EnvironmentError):
            env.env_dir_for_profile(profile)

    def test_known_compatibility_constraints_are_explicit(self) -> None:
        profiles = env.load_runtime_profiles()
        self.assertIn("numpy==1.26.4", profiles["medimage"].extra_requirements)
        self.assertIn("pandas==1.5.3", profiles["medimage"].extra_requirements)
        self.assertIn("6a761c4", profiles["pyradiomics"].requirement)
        self.assertEqual(profiles["pyradiomics"].source_commit, "6a761c4")

    def test_every_platform_lock_is_checksum_bound(self) -> None:
        macos_counts = {
            "medimage": 235,
            "mirp": 29,
            "pictologics": 18,
            "pyradiomics": 13,
            "zrad": 21,
        }
        for name, profile in env.load_runtime_profiles().items():
            keys = [lock.platform_key for lock in profile.environment_locks]
            self.assertEqual(keys, ["Darwin-arm64", "Linux-x86_64", "Windows-AMD64"])
            for lock in profile.environment_locks:
                path = env.repo_root() / lock.path
                requirements = sorted(
                    env.canonical_requirement(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
                self.assertEqual(path.name, f"{name}.txt")
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), lock.sha256
                )
                self.assertEqual(
                    hashlib.sha256("\n".join(requirements).encode()).hexdigest(),
                    lock.freeze_sha256,
                )
                if lock.platform_key == "Darwin-arm64":
                    self.assertEqual(len(requirements), macos_counts[name])

    def test_only_a_declared_unreadable_wheel_file_is_accepted(self) -> None:
        mirp = env.load_runtime_profiles()["mirp"]
        defect = (
            "Found 1 incompatibility\n"
            "The package `itk` is broken or incomplete (unable to read `WHEEL` "
            "file). Consider recreating the virtualenv."
        )
        self.assertTrue(env._only_known_metadata_defects(mirp, defect))
        self.assertFalse(
            env._only_known_metadata_defects(
                mirp,
                defect + "\nThe package `numpy` requires `scipy`, but it is missing",
            )
        )
        zrad = env.load_runtime_profiles()["zrad"]
        self.assertFalse(env._only_known_metadata_defects(zrad, defect))

    def test_requirement_names_compare_in_normalized_form(self) -> None:
        self.assertEqual(
            env.canonical_requirement("PyWavelets==1.9.0"), "pywavelets==1.9.0"
        )
        self.assertEqual(
            env.canonical_requirement("PyQt5_sip==12.19.0"), "pyqt5-sip==12.19.0"
        )
        self.assertEqual(
            env.canonical_requirement("Pkg.Name @ git+https://x/y.git@abc"),
            "pkg-name @ git+https://x/y.git@abc",
        )

    def test_recorded_environment_rejects_transitive_package_drift(self) -> None:
        freeze = ["adapter==1.0", "dependency==2.0"]
        freeze_sha256 = hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest()
        recorded = {
            "schema_version": 1,
            "profile": "adapter",
            "profile_fingerprint": "profile-hash",
            "python": "3.12.1",
            "python_implementation": "CPython",
            "distribution": "adapter",
            "version": "1.0",
            "distribution_metadata_version": "1.0",
            "freeze": freeze,
            "freeze_sha256": freeze_sha256,
        }
        observed = copy.deepcopy(recorded)
        observed["freeze"] = ["adapter==1.0", "dependency==2.1"]
        observed["freeze_sha256"] = hashlib.sha256(
            "\n".join(observed["freeze"]).encode("utf-8")
        ).hexdigest()

        with self.assertRaisesRegex(env.EnvironmentError, "freeze has drifted"):
            env._verify_recorded_environment(recorded, observed)

    def test_recorded_environment_rejects_tampered_freeze_checksum(self) -> None:
        recorded = {
            "schema_version": 1,
            "profile": "adapter",
            "profile_fingerprint": "profile-hash",
            "python": "3.12.1",
            "python_implementation": "CPython",
            "distribution": "adapter",
            "version": "1.0",
            "distribution_metadata_version": "1.0",
            "freeze": ["adapter==1.0"],
            "freeze_sha256": "not-the-freeze-hash",
        }

        with self.assertRaisesRegex(env.EnvironmentError, "checksum is invalid"):
            env._verify_recorded_environment(recorded, recorded)


if __name__ == "__main__":
    unittest.main()
