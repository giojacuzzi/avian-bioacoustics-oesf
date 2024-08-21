import argparse

if __name__ == "__main__":
    print('hello world')
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a custom classifier with BirdNET")
    parser.add_argument(
        "--i", default="train_data/", help="Path to training data folder. Subfolder names are used as labels."
    )
    parser.add_argument(
        "--o", default="checkpoints/custom/Custom_Classifier", help="Path to trained classifier model output."
    )
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, help="Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).")
    args = parser.parse_args()

    print(f"--i {args.i}")
    print(f"--o {args.o}")
    print(f"--autotune {args.autotune}")