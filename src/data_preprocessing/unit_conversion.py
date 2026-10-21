import pandas as pd

def rename_and_convert(df):
    df = df.copy()

    glucose_mmol       = df['LBXGLU'] / 18.0
    hba1c_mmolmol      = (df['LBXGH'] - 2.15) * 10.929
    hdl_mmol           = df['LBDHDD'] * 0.0259
    triglycerides_mmol = df['LBXSTR'] * 0.0113
    cholesterol_mmol   = df['LBXSCH'] * 0.0259
    uric_acid_umol     = df['LBXSUA'] * 59.48
    albumin_gL         = df['LBXSAL'] * 10
    creatinine_umol    = df['LBXSCR'] * 88.4
    alt_UL             = df['LBXSATSI']
    sodium_mmolL       = df['LBXSNASI']
    bun_mmolL          = df['LBXSBU'] * 0.357
    ggt_UL             = df['LBXSGTSI']
    ast_UL             = df['LBXSASSI']

    result = pd.DataFrame({
        'PatientID':                df['SEQN'],
        'Age':                      df['RIDAGEYR'],
        'Gender':                   df['RIAGENDR'].map({1: 'Male', 2: 'Female'}),
        'Triglycerides':            triglycerides_mmol.round(2),
        'TotalCholesterol':         cholesterol_mmol.round(2),
        'UricAcid':                 uric_acid_umol.round(1),
        'Albumin':                  albumin_gL.round(1),
        'Creatinine':               creatinine_umol.round(1),
        'Alanine Aminotransferase': alt_UL,
        'FastingGlucose':           glucose_mmol.round(2),
        'HDLCholesterol':           hdl_mmol.round(2),
        'Glycated Haemoglobin':     hba1c_mmolmol.round(1),
        'Sodium':                   sodium_mmolL.round(1),
        'BUN':                      bun_mmolL.round(2),
        'GGT':                      ggt_UL.round(1),
        'AST':                      ast_UL.round(1),
        'Cycle':                    df['CYCLE'],
    })
    return result