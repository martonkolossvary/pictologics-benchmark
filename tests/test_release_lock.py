from __future__ import annotations

import hashlib
import json
import unittest

from bench import release_lock


class ReleaseLockTests(unittest.TestCase):
    def test_every_bound_file_matches_the_release_lock(self) -> None:
        # A deliberate change needs: python -m bench.release_lock --write
        release_lock.verify()

    def test_the_current_release_is_accepted_for_submissions(self) -> None:
        digest = hashlib.sha256(release_lock.LOCK.read_bytes()).hexdigest()
        accepted = json.loads(
            (release_lock.REPO / "submissions" / "accepted_releases.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(digest, accepted["release_lock_sha256"])


if __name__ == "__main__":
    unittest.main()
