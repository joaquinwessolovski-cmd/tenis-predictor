from train_model import load_data, build_dataset

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print("Building dataset and populating EloHistory...")
    build_dataset(df)
    print("Done!")
