import os

PROJECT_BASE_PATH = os.path.abspath(
    "C:\\Users\\cm03009\\Documents\\OneShotObjectDetection"
)
query_dir = os.path.join(PROJECT_BASE_PATH, "Queries")
test_dir = os.path.join(PROJECT_BASE_PATH, "Test")
results_dir = os.path.join(PROJECT_BASE_PATH, "Results")

os.makedirs(query_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
