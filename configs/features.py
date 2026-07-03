ID_COLUMNS = ["encounter_id", "patient_nbr"]
TARGET = "readmitted_binary"

NUMERICAL_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

DEMOGRAPHIC_FEATURES = [
    "race", 
    "gender", 
    "age"
]

CLINICAL_FEATURES = [
    "medical_specialty_group",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed"
]

MEDICATION_FEATURES = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", 
    "glimepiride", "acetohexamide", "glipizide", "glyburide", 
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", 
    "miglitol", "troglitazone", "tolazamide", "examide", 
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", 
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
]
