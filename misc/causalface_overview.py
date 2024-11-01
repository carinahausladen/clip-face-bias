import random

from datasets.causalface import CausalFaceDataset

available_seeds = CausalFaceDataset(subset="age").available_seeds

random.seed(42)
random.shuffle(available_seeds)

for i in range(20):
    for subset in ["age", "smiling", "lighting", "pose"]:
        seed = available_seeds[i]
        ds = CausalFaceDataset(subset=subset)
        ds.create_seed_overview_image(seed=seed,  show=False)
        print(f"Saved {seed} {subset}")