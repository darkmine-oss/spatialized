import subprocess
import sys


def test_paper_like_experiment_smoke(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "examples/paper_like_experiment.py",
            "--output-dir",
            str(tmp_path),
            "--size",
            "24",
            "--samples-per-class",
            "6",
            "--unsupervised-samples",
            "24",
            "--trees",
            "30",
            "--chunk-size",
            "128",
            "--clusters",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "predicted_classes.tif" in result.stdout
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "prediction_entropy.tif").exists()
