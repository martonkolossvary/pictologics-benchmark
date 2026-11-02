from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import host_state


def _supply(root: Path, name: str, kind: str, online: str | None = None) -> None:
    folder = root / name
    folder.mkdir()
    (folder / "type").write_text(kind + "\n", encoding="ascii")
    if online is not None:
        (folder / "online").write_text(online + "\n", encoding="ascii")


class LinuxPowerStateTests(unittest.TestCase):
    def probe(self, build) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build(root)
            with mock.patch.object(host_state.shutil, "which", return_value=None):
                return host_state.linux_power_state(root)

    def test_a_desktop_without_a_battery_is_on_mains_power(self) -> None:
        state = self.probe(lambda root: None)
        self.assertEqual(state["power_source"], "AC Power")
        self.assertEqual(state["low_power_mode"], 0)
        self.assertIsNone(state["sleep_assertion"])

    def test_a_laptop_reports_its_charger(self) -> None:
        def plugged(root: Path) -> None:
            _supply(root, "AC", "Mains", "1")
            _supply(root, "BAT0", "Battery")

        def unplugged(root: Path) -> None:
            _supply(root, "AC", "Mains", "0")
            _supply(root, "BAT0", "Battery")

        self.assertEqual(self.probe(plugged)["power_source"], "AC Power")
        self.assertEqual(self.probe(unplugged)["power_source"], "Battery Power")


if __name__ == "__main__":
    unittest.main()
