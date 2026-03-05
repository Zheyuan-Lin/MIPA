PHENOTYPES = ["c_diff_complication", "c_diff_past", "dementia", "depression", "diabetes_type_1",
         "diabetes_type_2", "hfref", "hfpef", "hypertension", "sle",
         "metastatic_cancer", "obesity","rheumatoid_arthritis", "alcohol_abuse", "vte_complication", "vte_past"]


ETP_ABBREVIATIONS = {
    "c_diff_complication": ["cdiff", "c. diff", "clostridium difficile", "c difficile", "c diff"],
    "c_diff_past": ["cdiff", "c. diff", "clostridium difficile", "c difficile", "c diff"],
    "dementia": ["dementia", "dement", "demented", "dementedness"],
    "depression": ["depression", "depress", "depressed", "depressive"],
    "diabetes_type_1": ["diabetes_type_1", "diabetes type 1", "type 1 diabetes", "type1 diabetes", "type1diabetes", "type 1 diabetic", "type1 diabetic", "dm1", "dm i", "dm i diabetic"],
    "diabetes_type_2": ["diabetes_type_2", "diabetes type 2", "type 2 diabetes", "type2 diabetes", "type2diabetes", "type 2 diabetic", "type2 diabetic", "dm2", "dm ii", "dm ii diabetic"],
    "hfref": ["hfref", "hf ref", "heart failure reduced ejection fraction", "heart failure ref", "hf reduced ejection fraction", "hfr ef", "hfr ef"],
    "hfpef": ["hfpef", "hf pef", "heart failure preserved ejection fraction", "heart failure pef", "hf preserved ejection fraction", "hfp ef", "hfp ef"],
    "hypertension": ["hypertension", "high blood pressure", "high bp", "high blood presure", "highbp", "htn"],
    "sle": ["sle", "systemic lupus erythematosus", "lupus", "systemic lupus", "systemic lupus erythematosis"],
    "rheumatoid_arthritis": ["ra", "rhum arthritis", "arthritis"],
    "metastatic_cancer": ["metastatic_cancer", "metastatic cancer", "metastatic cancers", "metastatic tumor", "metastatic tumors", "met cancer", "metastatic ca"],
    "obesity": ["obesity", "obese", "obes", "obesities"],
    "roh_abuse": ["roh_abuse", "roh abuse", "alcohol abuse", "alcoholism", "alcoholic", "alcohol addiction", "etoh abuse"],
    "vte_past": [
        # General VTE terms
        "vte", "venous thromboembolism", "venous thromboembolic", "venous thromboembolisms",
        # DVT related
        "dvt", "deep vein thrombosis", "deep venous thrombosis", "deep vein thrombus", "deep venous thrombus",
        # PE related
        "pe", "pulmonary embolism", "pulmonary embolus", "pulmonary emboli", "pulmonary thromboembolism",
        # Splanchnic VTE
        "svt", "splanchnic venous thrombosis", "portal vein thrombosis", "pvt",
        "mesenteric vein thrombosis", "mvt", "splenic vein thrombosis",
        "hepatic vein thrombosis", "budd-chiari", "budd chiari",
        # CVST related
        "cvst", "cerebral venous sinus thrombosis", "cerebral sinus thrombosis",
        "sinus thrombosis", "dural sinus thrombosis", "cerebral venous thrombosis"
    ],
    "vte_complication": [
        # General VTE terms
        "vte", "venous thromboembolism", "venous thromboembolic", "venous thromboembolisms",
        # DVT related
        "dvt", "deep vein thrombosis", "deep venous thrombosis", "deep vein thrombus", "deep venous thrombus",
        # PE related
        "pe", "pulmonary embolism", "pulmonary embolus", "pulmonary emboli", "pulmonary thromboembolism",
        # Splanchnic VTE
        "svt", "splanchnic venous thrombosis", "portal vein thrombosis", "pvt",
        "mesenteric vein thrombosis", "mvt", "splenic vein thrombosis",
        "hepatic vein thrombosis", "budd-chiari", "budd chiari",
        # CVST related
        "cvst", "cerebral venous sinus thrombosis", "cerebral sinus thrombosis",
        "sinus thrombosis", "dural sinus thrombosis", "cerebral venous thrombosis"
        ],
    "metastatic cancer": ["metastatic cancer", "metastatic cancers", "metastatic tumor", "metastatic tumors", "met cancer", "metastatic ca"]
}


PHENOTYPES_ETP_KEYS = {
    "c_diff_past": [
        "c_diff_past",
        "recurrent",
        "prior_treatment",
        "prior_fmt",
        "prior_complications"
    ],

    "c_diff_complication": [
        "c_diff_complication",
        "symptoms",
        "diagnostic_tests",
        "colonoscopy",
        "treatment",
        "complications",
        "fmt_needed",
        "nosocomial"
    ],

    "diabetes_type_2": [
        "diabetes_type_2", "metformin", "sulfonylureas", "glp1_agonists",
        "dpp4_inhibitors", "sglt2_inhibitors", "insulin", "neuropathy",
        "retinopathy", "nephropathy", "hba1c"
    ],
    "diabetes_type_1": [
        "diabetes_type_1", "insulin_therapy_from_diagnosis", "insulin_therapy_any",
        "autoantibodies", "ketoacidosis", "no_oral_hypoglycemics", "neuropathy",
        "retinopathy", "nephropathy"
    ],
    "dementia": [
        "dementia", "cognitive_decline", "memory_loss", "disorientation",
        "behavioral_changes", "cholinesterase_inhibitors", "memantine",
        "neuroimaging_findings", "functional_decline"
    ],
    "depression": [
        "depression", "mood_symptoms", "sleep_disturbances", "appetite_changes",
        "antidepressants", "psychotherapy", "fatigue", "suicide_risk"
    ],
    "hypertension": [
        "hypertension", "blood_pressure_readings", "antihypertensive_medications",
        "end-organ_damage"
    ],
    "hfpef": [
        "hfpef", "preserved_EF", "diastolic_dysfunction", "congestion_symptoms",
        "elevated_filling_pressures", "bnp_levels", "diuretics"
    ],
    "hfref": [
        "hfref", "reduced_EF", "congestion_symptoms", "bnp_levels", "beta_blockers",
        "arni", "diuretics", "acei_arb", "aldosterone_antagonists"
    ],
    "sle": [
        "sle", "ANA", "specific_antibodies", "complement_levels", "skin_manifestations",
        "joint_symptoms", "renal_involvement", "serositis", "dmards", "steroids"
    ],
    "metastatic_cancer": [
        "metastatic_cancer", "primary_tumor_site", "metastasis_sites",
        "biopsy_results", "oncological_treatments"
    ],
    "rheumatoid_arthritis": [
        "rheumatoid_arthritis", "joint_symptoms", "morning_stiffness",
        "serological_markers", "imaging_findings", "cdmards", "biologics",
        "steroids", "extra-articular_manifestations"
    ],
    "obesity": [
        "obesity", "BMI", "waist_circumference", "comorbid_conditions",
        "bariatric_surgery", "medication_for_obesity"
    ],
    "alcohol_abuse": [
        "alcohol_abuse", "consumption_patterns", "withdrawal_symptoms",
        "liver_abnormalities", "comorbid_conditions", "treatment_for_addiction",
        "medications"
    ],
    "vte_past": [
        "vte_past", "dvt_past", "pe_past", "svt_past", "cvst_past", "recurrent",
        "anticoagulation", "ivc_filter", "thrombophilia"
    ],

    "vte_complication": [
        "vte_complication", "dvt_current", "pe_current", "svt_current", "cvst_current", "vte_symptoms",
        "diagnostic_tests", "anticoagulation", "interventions", "nosocomial"
    ]
}

PHENOTYPES_ETP_SCORES = {
  "c_diff_past": {
        "c_diff_past": 2,
        "recurrent": 2,
        "prior_treatment": 1,
        "prior_fmt": 1,
        "prior_complications": 1
  },

  "c_diff_complication": {
        "c_diff_complication": 2,
        "symptoms": 1,
        "diagnostic_tests": 2,
        "colonoscopy": 1,
        "treatment": 2,
        "complications": 1,
        "fmt_needed": 1,
        "nosocomial": 1
  },

  "diabetes_type_2": {
    "diabetes_type_2": 2,
    "metformin": 2,
    "sulfonylureas": 2,
    "glp1_agonists": 0,
    "dpp4_inhibitors": 2,
    "sglt2_inhibitors": 0,
    "insulin": 1,
    "neuropathy": 1,
    "retinopathy": 1,
    "nephropathy": 1,
    "gluose_levels": 1,
    "hba1c": 1
  },

  "diabetes_type_1": {
    "diabetes_type_1": 2,
    "insulin_therapy_from_diagnosis": 2,
    "insulin_therapy_any": 1,
    "autoantibodies": 1,
    "ketoacidosis": 2,
    "no_oral_hypoglycemics": 2,
    "neuropathy": 1,
    "retinopathy": 1,
    "nephropathy": 1
  },

  "dementia": {
    "dementia": 2,
    "cognitive_decline": 1,
    "memory_loss": 1,
    "disorientation": 1,
    "behavioral_changes": 1,
    "cholinesterase_inhibitors": 2,
    "memantine": 2,
    "neuroimaging_findings": 1,
    "functional_decline": 2
  },
  "depression": {
    "depression": 2,
    "mood_symptoms": 1,
    "sleep_disturbances": 1,
    "appetite_changes": 1,
    "antidepressants": 2,
    "psychotherapy": 1,
    "fatigue": 1,
    "suicide_risk": 1
  },
  "hypertension": {
    "hypertension": 2,
    "blood_pressure_readings": 2,
    "antihypertensive_medications": 2,
    "end-organ_damage": 1
  },
  "hfpef": {
    "hfpef": 2,
    "preserved_EF": 2,
    "diastolic_dysfunction": 2,
    "congestion_symptoms": 1,
    "elevated_filling_pressures": 1,
    "bnp_levels": 1,
    "diuretics": 1
  },

  "hfref": {
    "hfref": 2,
    "reduced_EF": 2,
    "congestion_symptoms": 1,
    "bnp_levels": 1,
    "beta_blockers": 1,
    "arni": 1,
    "diuretics": 1,
    "acei_arb": 1,
    "aldosterone_antagonists": 1
  },
  "sle": {
    "sle": 2,
    "ANA": 2,
    "specific_antibodies": 2,
    "complement_levels": 2,
    "skin_manifestations": 1,
    "joint_symptoms": 1,
    "renal_involvement": 1,
    "serositis": 1,
    "dmards": 2,
    "steroids": 1
  },
  "metastatic_cancer": {
    "metastatic_cancer": 2,
    "primary_tumor_site": 1,
    "metastasis_sites": 2,
    "biopsy_results": 1,
    "oncological_treatments": 1
  },
  "rheumatoid_arthritis": {
    "rheumatoid_arthritis": 2,
    "joint_symptoms": 1,
    "morning_stiffness": 1,
    "serological_markers": 2,
    "imaging_findings": 1,
    "cdmards": 2,
    "biologics": 2,
    "steroids": 1,
    "extra-articular_manifestations": 1
  },
  "obesity": {
    "obesity": 2,
    "BMI": 2,
    "waist_circumference": 1,
    "comorbid_conditions": 1,
    "bariatric_surgery": 2,
    "medication_for_obesity": 1
  },
  "alcohol_abuse": {
    "alcohol_abuse": 2,
    "consumption_patterns": 1,
    "withdrawal_symptoms": 1,
    "liver_abnormalities": 1,
    "comorbid_conditions": 1,
    "treatment_for_addiction": 1,
    "medications": 2
  },
  "vte_past": {
        "phenotype": 1,
        "dvt_past": 0,
        "pe_past": 0,
        "svt_past": 0,
        "cvst_past": 0,
        "recurrent": 1,
        "anticoagulation": 2,
        "ivc_filter": 2,
        "thrombophilia": 2
  },

  "vte_complication": {
        "phenotype": 1,
        "dvt_current": 0,
        "pe_current": 0,
        "svt_current": 0,
        "cvst_current": 0,
        "vte_symptoms": 1,
        "diagnostic_tests": 2,
        "anticoagulation": 2,
        "interventions": 1,
        "nosocomial": 1
  }
}
