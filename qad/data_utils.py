from datasets import load_dataset

def get_tulu_train_val():
    DATASET = "allenai/tulu-3-sft-mixture"
    splits = load_dataset(DATASET, split="train").train_test_split(test_size=0.1, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    return train_dataset, eval_dataset
