"""
Basic tests for the factuality evaluation project.
"""

import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class TestProjectStructure(unittest.TestCase):
    """Test basic project structure."""

    def test_directories_exist(self):
        """Test that essential directories exist."""
        essential_dirs = [
            "src", "data", "results", "config", "experiments"
        ]

        for dir_name in essential_dirs:
            with self.subTest(directory=dir_name):
                self.assertTrue(
                    Path(dir_name).exists(),
                    f"Directory {dir_name} should exist"
                )

    def test_config_files_exist(self):
        """Test that essential config files exist."""
        essential_files = [
            "config/default.yaml",
            "requirements.txt",
            "setup.sh"
        ]

        for file_name in essential_files:
            with self.subTest(file=file_name):
                self.assertTrue(
                    Path(file_name).exists(),
                    f"File {file_name} should exist"
                )


if __name__ == "__main__":
    unittest.main()
