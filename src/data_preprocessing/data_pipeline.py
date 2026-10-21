import pandas as pd
from functools import reduce
import os
from src.data_preprocessing.unit_conversion import rename_and_convert
from src.data_preprocessing.labels import derive_diabetes_label

RAW_DATA_DIR = 'data/raw'
CYCLE_FOLDERS = ['1-2', '3-4', '5-6', '7-8', '9-10', '11-12', '13-14', '15-16', '17-20', '21-23']

def load_cycle(folder_name):
    folder_path = os.path.join(RAW_DATA_DIR, folder_name)
    demo = pd.read_sas(os.path.join(folder_path, 'DEMO.xpt'), format='xport')[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
    bio_raw = pd.read_sas(os.path.join(folder_path, 'BIOPRO.xpt'), format='xport')
    creatinine_col = 'LBXSCR' if 'LBXSCR' in bio_raw.columns else 'LBDSCR'
    bio = bio_raw[['SEQN', 'LBXSTR', 'LBXSCH', 'LBXSUA', 'LBXSAL', creatinine_col,'LBXSATSI', 'LBXSNASI', 'LBXSBU', 'LBXSGTSI', 'LBXSASSI']].copy()
    bio = bio.rename(columns={creatinine_col: 'LBXSCR'})
    glu = pd.read_sas(os.path.join(folder_path, 'GLU.xpt'), format='xport')[['SEQN', 'LBXGLU']]
    hdl_raw = pd.read_sas(os.path.join(folder_path, 'HDL.xpt'), format='xport')
    possible_hdl = ['LBDHDD', 'LBDHDL', 'LBXHDD']
    hdl_col = next((c for c in possible_hdl if c in hdl_raw.columns), None)
    if hdl_col is None:
        raise KeyError(f"No known HDL column in {folder_path}: {hdl_raw.columns.tolist()}")
    hdl = hdl_raw[['SEQN', hdl_col]].rename(columns={hdl_col: 'LBDHDD'})
    ghb = pd.read_sas(os.path.join(folder_path, 'GHB.xpt'), format='xport')[['SEQN', 'LBXGH']]
    dfs = [demo, bio, glu, hdl, ghb]
    merged = reduce(lambda l, r: pd.merge(l, r, on='SEQN', how='left'), dfs)
    merged['CYCLE'] = folder_name
    return merged

def build_dataset():
    all_cycles = []
    for folder in CYCLE_FOLDERS:
        try:
            df = load_cycle(folder)
            print(f"Loaded {folder}: {df.shape}")
            all_cycles.append(df)
        except Exception as e:
            print(f"FAILED to load {folder}: {e}")

    nhanes_combined = pd.concat(all_cycles, ignore_index=True)
    nhanes_combined.to_csv('data/processed/merged_raw.csv', index=False)
    print(f"\nCombined shape: {nhanes_combined.shape}")
    print(f"Duplicate SEQNs: {nhanes_combined['SEQN'].duplicated().sum()}")
    df_clean = nhanes_combined.dropna()
    df_clean.to_csv('data/processed/merged_clean.csv', index=False)
    print(f"Clean shape: {df_clean.shape}")
    df_uk = rename_and_convert(df_clean)
    df_uk = derive_diabetes_label(df_uk)
    df_uk.to_csv('data/processed/labeled_dataset.csv', index=False)
    print("\nFinal class distribution:")
    print(df_uk['DiabetesStatus'].value_counts())
    return df_uk

if __name__ == '__main__':
    build_dataset()