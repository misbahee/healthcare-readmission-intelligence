from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

cols_to_encode = [
        'race', 'payer_code', 'diag_1_group', 'diag_2_group', 'diag_3_group', 
        'medical_specialty_group','admission_type_id','discharge_disposition_id', 
        'admission_source_id', 'gender', 'change', 'diabetesMed', 
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
        'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 
        'rosiglitazone', 'acarbose', 'glyburide-metformin', 'glipizide-metformin',
        'A1C_clinical_response'
]

cols_to_scale = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses', 'total_prior_visits'
]

def get_preprocessor():
    

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'one_hot_encoder',
                OneHotEncoder(
                    drop='first',
                    handle_unknown='ignore', 
                    sparse_output=False
                ), cols_to_encode
            ),
            (
                'scaler', 
                StandardScaler(), 
                cols_to_scale
            )
        ],
        remainder='passthrough'
    )

    return preprocessor