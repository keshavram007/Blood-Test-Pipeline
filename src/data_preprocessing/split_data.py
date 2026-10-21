import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    df = pd.read_csv('data/processed/labeled_dataset.csv')

    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['DiabetesStatus'], random_state=42)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['DiabetesStatus'], random_state=42)

    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)

    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print(train_df['DiabetesStatus'].value_counts(normalize=True).round(3))
    return train_df, val_df, test_df


if __name__ == '__main__':
    split_dataset()