from pathlib import Path


def test_cpp_project_files_exist():
    root = Path(__file__).parents[1]
    assert (root / "csrc" / "CMakeLists.txt").exists()
    assert (root / "csrc" / "include" / "engine.hpp").exists()
