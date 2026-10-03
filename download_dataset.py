from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vidit031/isl-isolated-40words",
    repo_type="dataset",
    local_dir="data/isl-isolated-40words"
)

print("Dataset downloaded successfully.")