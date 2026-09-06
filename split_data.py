import splitfolders

splitfolders.ratio(
    "dataset",
    output="final_dataset",
    seed=42,
    ratio=(0.7, 0.2, 0.1)
)

print("✅ Dataset split completed")