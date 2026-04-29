import context  # noqa: F401 E402
from scripts.runs.run_maves import run_maves

datasets = [
    "MC4R",
    "HXK4",
    "PTEN",
    "SRC",
]

if __name__ == "__main__":
    run_maves(None, datasets, "./config/msalde_paper.yaml", 50)

