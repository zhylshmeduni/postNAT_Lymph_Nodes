import re
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df_postnat = pd.read_excel('D:\\Python_project\\Macrosection\\Lymph_Nodes\Files\\Review1026\\FUSCC_trainvaltest_postNAT0114.xlsx', sheet_name='FUSCC主表')
df_clinical = pd.read_excel('D:\\Python_project\\Macrosection\\Lymph_Nodes\\Files\\Review1026\\FUSCC_Clinical.xlsx', sheet_name='Sheet 1')

df_postnat['patient_id'] = df_postnat['patient_id'].astype(str)
df_clinical['Patho_ID'] = df_clinical['Patho_ID'].astype(str)

df_merged = pd.merge(
    df_postnat,
    df_clinical[['Patho_ID', 'Subtype', 'ypN_Stage', 'Histology']],
    left_on='patient_id',
    right_on='Patho_ID',
    how='left'
)

df_patients = df_merged[['patient_id', 'Subtype', 'ypN_Stage', 'Histology']].drop_duplicates().reset_index(drop=True)
df_patients.fillna('Unknown', inplace=True)

df_patients['Histology_Binary'] = df_patients['Histology'].apply(lambda x: 'ILC' if x == 'ILC' else 'Non_ILC')

df_patients['stratify_key'] = (
    df_patients['Subtype'].astype(str) + "_" +
    df_patients['ypN_Stage'].astype(str) + "_" +
    df_patients['Histology_Binary'].astype(str)
)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

for train_idx, val_idx in splitter.split(df_patients, df_patients['stratify_key']):
    train_patients = df_patients.iloc[train_idx]['patient_id'].values
    val_patients = df_patients.iloc[val_idx]['patient_id'].values

patient_group_map = {}
for p_id in train_patients:
    patient_group_map[p_id] = 'training'
for p_id in val_patients:
    patient_group_map[p_id] = 'validation'

df_postnat_out = df_postnat.copy()
df_postnat_out['group'] = df_postnat_out['patient_id'].map(patient_group_map)
df_postnat_out.to_excel('FUSCC_trainvaltest_postNAT0114_split.xlsx', index=False)

df_clinical_out = df_clinical.copy()
df_clinical_out['group'] = df_clinical_out['Patho_ID'].map(patient_group_map)
df_clinical_out.to_excel('FUSCC_Clinical_with_group.xlsx', index=False)