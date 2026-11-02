from __future__ import annotations

import csv
import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReproducibilityPackageTests(unittest.TestCase):
    def test_external_input_manifest_is_complete_unique_and_data_only(self) -> None:
        manifest = json.loads(
            (ROOT / "reproducibility/inputs/manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            manifest["repositories"]["ibsi_data_sets"]["commit"],
            "6da96021bc91faf4c0cb7fd7fa56a4225d2064a8",
        )
        self.assertEqual(
            manifest["repositories"]["ibsi2_reference"]["commit"],
            "5404579fb3e0d17e8db421f0e82d64ce2432ec03",
        )
        entries = manifest["files"]
        self.assertEqual(len(entries), 371)
        self.assertEqual(
            Counter(entry["component"] for entry in entries),
            {
                "ibsi1": 2,
                "ibsi2-phase1": 50,
                "ibsi2-phase2": 3,
                "ibsi2-phase3": 306,
                "licenses": 10,
            },
        )
        destinations = [entry["destination"] for entry in entries]
        self.assertEqual(len(destinations), len(set(destinations)))
        for entry in entries:
            self.assertTrue(entry["source"])
            self.assertTrue(entry["destination"].startswith("data/"))
            self.assertRegex(entry["sha256"], SHA256)
            self.assertGreater(entry["bytes"], 0)

    def test_feature_surface_contract_is_checksum_bound_and_parseable(self) -> None:
        stem = "reproducibility/contracts/adapter_feature_surface"
        contract = ROOT / f"{stem}.csv"
        metadata = json.loads((ROOT / f"{stem}.json").read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(contract.read_bytes()).hexdigest(),
            metadata["contract_sha256"],
        )
        with contract.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), metadata["rows"])
        self.assertEqual(set(rows[0]), {"adapter", "family", "native_feature_names"})
        self.assertEqual(
            {row["adapter"] for row in rows},
            {"medimage", "mirp", "pictologics", "pyradiomics", "zrad"},
        )

    def test_project_uses_the_same_apache_license_identifier_as_pictologics(
        self,
    ) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn('license = "Apache-2.0"', pyproject)
        self.assertTrue((ROOT / "NOTICE").is_file())


if __name__ == "__main__":
    unittest.main()
