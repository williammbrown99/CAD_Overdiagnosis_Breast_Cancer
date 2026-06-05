# Import pandas for tabular data manipulation
import pandas as pd
# Import numpy for fast mathematical operations and NaN handling
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
# Import functions to split data and tune hyperparameters randomly
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# Import scoring metrics to evaluate the model's performance
from sklearn.metrics import confusion_matrix
# Import LabelEncoder to turn text categories (like 'Positive') into numbers (like 1)
from sklearn.preprocessing import LabelEncoder
# Import the core XGBoost algorithm
from xgboost import XGBClassifier
# Suppress unnecessary terminal warnings to keep output clean
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA & GOLDEN COHORT FILTER
# ==========================================
filename = "INPUT/SEER_10436_in_situ_included.csv"
print(f"Loading {filename}...")
df = pd.read_csv(filename, low_memory=False)

df['Survival months'] = pd.to_numeric(df['Survival months'], errors='coerce')
df['Year of diagnosis'] = pd.to_numeric(df['Year of diagnosis'], errors='coerce')

# VITAL FILTER 1 & 2: Active survival > 0 and exclude post-mortem diagnoses
df = df[df['Survival months'] > 0] 
if 'Type of Reporting Source' in df.columns:
    autopsy_mask = df['Type of Reporting Source'].astype(str).str.contains('autopsy|death certificate', case=False, na=False)
    df = df[~autopsy_mask]

# Era Constraints
df = df[(df['Year of diagnosis'] >= 1990) & (df['Year of diagnosis'] <= 2016)].copy()

# Baseline Demographic Filters
df_base = df[
    (df['Marital status at diagnosis'] != 'Unknown') & 
    (~df['Median household income inflation adj to 2023'].str.contains('Unknown', na=False, case=False))
].copy()

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
size_cols = ['Tumor Size Summary (2016+)', 'CS tumor size (2004-2015)', 
             'EOD 10 - size (1988-2003)', 'EOD 4 - size (1983-1987)']

def clean_size(val):
    try:
        v = float(val)
        if v == 990: return 0.5
        return v if 0 <= v <= 400 else np.nan
    except: return np.nan

for col in size_cols:
    if col in df_base.columns:
        df_base[col] = df_base[col].apply(clean_size)
df_base['Master_Tumor_Size'] = df_base[size_cols].bfill(axis=1).iloc[:, 0]

grade_cols = ['Grade Recode (thru 2017)', 'Grade Clinical (2018+)', 'Grade Pathological (2018+)']

def clean_grade(val):
    v = str(val).lower()
    if any(x in v for x in ['3', '4', 'poor', 'undiff', 'iii', 'iv']): return 3
    if any(x in v for x in ['2', 'mod', 'ii']): return 2
    if any(x in v for x in ['1', 'well', 'grade i']) and 'ii' not in v: return 1
    return np.nan

df_base['Master_Grade'] = np.nan
for col in grade_cols:
    if col in df_base.columns:
        df_base['Master_Grade'] = df_base['Master_Grade'].fillna(df_base[col].apply(clean_grade))

stage_cols = ['Summary stage 2000 (1998-2017)', 'SEER historic stage A (1973-2015)', 'Combined Summary Stage with Expanded Regional Codes (2004+)']

def clean_stage(val):
    v = str(val).lower()
    if 'distant' in v: return 3
    if 'regional' in v: return 2
    if 'localized' in v: return 1
    if 'in situ' in v: return 0
    return np.nan

df_base['Master_Stage'] = np.nan
for col in stage_cols:
    if col in df_base.columns:
        df_base['Master_Stage'] = df_base['Master_Stage'].fillna(df_base[col].apply(clean_stage))

age_candidates = ['Age recode with single ages and 90+', 'Age recode with single ages and 85+', 'Age recode with <1 year olds']
actual_age_col = next((c for c in age_candidates if c in df_base.columns), None)
df_base['Age_Numeric'] = df_base[actual_age_col].apply(lambda x: int(str(x).split()[0]) if pd.notnull(x) else np.nan)

# Interactions & History
df_base['Age_Stage_Interaction'] = df_base['Age_Numeric'] * df_base['Master_Stage']
df_base['Age_Tumor_Size_Interaction'] = df_base['Age_Numeric'] * df_base['Master_Tumor_Size']

if 'Sequence number' in df_base.columns:
    df_base['Has_Previous_History'] = df_base['Sequence number'].apply(lambda x: 0 if 'One primary only' in str(x) else 1)
else:
    df_base['Has_Previous_History'] = 0

# THE FIREWALL: Whitelist
protected_cols = [
    'Target', 'Age_Numeric', 'Master_Tumor_Size', 'Master_Grade', 'Master_Stage', 
    'ER Status Recode Breast Cancer (1990+)', 'PR Status Recode Breast Cancer (1990+)',
    'Has_Previous_History', 'Age_Stage_Interaction', 'Age_Tumor_Size_Interaction', 
    'COD to site recode', 'Survival months',
    'Histology recode - broad groupings', 'Site recode - rare tumors'
]

df_base = df_base[[c for c in protected_cols if c != 'Target' and c in df_base.columns]]

# Encode Categoricals & Handle Receptor NaNs
for col in ['ER Status Recode Breast Cancer (1990+)', 'PR Status Recode Breast Cancer (1990+)']:
    if col in df_base.columns:
        df_base[col] = df_base[col].replace(['Borderline/Unknown', 'Recode not available', 'Unknown', 'Blank(s)'], np.nan)

encode_cols = [c for c in df_base.columns if not pd.api.types.is_numeric_dtype(df_base[c]) and c not in ['COD to site recode']]
for col in encode_cols:
    df_base[col] = LabelEncoder().fit_transform(df_base[col].fillna("Missing").astype(str))

print("Final number of Patients: ")
print(len(df_base))

# ==========================================
# 3. ISOLATE, SPLIT, AND LOCK THE COHORTS
# ==========================================
print("\n" + "="*80)
print("   PRE-SPLITTING TO GUARANTEE CUMULATIVE DATA INTEGRITY   ")
print("="*80)

cond_base_agg = (df_base['COD to site recode'] == 'Breast') & (df_base['Survival months'] < 60)
df_base_agg = df_base[cond_base_agg].copy()
df_base_agg['Target'] = 0 
train_base_agg, test_base_agg = train_test_split(df_base_agg, test_size=0.2, random_state=42)

cond_ext_agg = (df_base['COD to site recode'] == 'Breast') & (df_base['Survival months'] >= 60)
df_ext_agg = df_base[cond_ext_agg].copy()
df_ext_agg['Target'] = 0
train_ext_agg, test_ext_agg = train_test_split(df_ext_agg, test_size=0.2, random_state=42)

print(f"Locked BASE Aggressive Test Set (< 5 yrs): {len(test_base_agg)} patients (The Original 494)")
print(f"Locked EXTENDED Aggressive Pool (>= 5 yrs): Ready for incremental 'unlocking'.")

# ==========================================
# 4. TEMPORAL EXPERT MODELS (ALL THRESHOLDS)
# ==========================================
thresholds_yrs = [5, 6, 7, 8, 9, 10]
optimal_ratios = []
kmf_aggressive_dict = {}

for yrs in thresholds_yrs:
    thresh_mos = yrs * 12 
    
    print("\n" + "="*80)
    print(f"   TRAINING & PLOTTING EXPERT MODEL: {yrs}-YEAR THRESHOLD   ")
    print("="*80)
    
    # Assemble current cohorts
    train_agg_current = pd.concat([train_base_agg, train_ext_agg[train_ext_agg['Survival months'] < thresh_mos]])
    test_agg_current = pd.concat([test_base_agg, test_ext_agg[test_ext_agg['Survival months'] < thresh_mos]])
    
    cond_over = (df_base['COD to site recode'] != 'Breast') & (df_base['Survival months'] >= thresh_mos)
    df_over = df_base[cond_over].copy()
    df_over['Target'] = 1 
    train_over, test_over = train_test_split(df_over, test_size=0.2, random_state=42)
    
    train_full = pd.concat([train_agg_current, train_over])
    test_full = pd.concat([test_agg_current, test_over])
    
    # PRESERVE SURVIVAL DATA FOR KAPLAN-MEIER VALIDATION LATER
    df_eval = test_full[['Survival months', 'COD to site recode', 'Target']].copy()
    
    # Strip cheat columns
    X_train = train_full.drop(columns=['Target', 'COD to site recode', 'Survival months'])
    y_train = train_full['Target']
    X_test = test_full.drop(columns=['Target', 'COD to site recode', 'Survival months'])
    y_test = test_full['Target']
    
    print(f"[!] Running Hyperparameter Tuning...")
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    xgb_base = XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42)
    param_distributions = {
        'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2], 'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    random_search = RandomizedSearchCV(xgb_base, param_distributions=param_distributions, n_iter=20, scoring='f1', cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    y_probs = best_xgb.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # ADVISOR TEST: SLIDING SCALE (100:0 down to 0:100, step 1)
    # ---------------------------------------------------------
    print(f"[!] Running Sliding Scale Optimization for {yrs}-Year Threshold...")
    results_list = []

    for w_fp in range(100, -1, -1):
        w_fn = 100 - w_fp
        best_thresh = 0.50
        min_cost = float('inf')
        best_cm = None
        
        for thresh in np.arange(0.01, 1.00, 0.01):
            temp_pred = (y_probs >= thresh).astype(int)
            cm_temp = confusion_matrix(y_test, temp_pred)
            
            if cm_temp.shape == (2, 2):
                tn, fp, fn, tp = cm_temp.ravel()
                cost = (fp * w_fp) + (fn * w_fn)
                
                if cost < min_cost:
                    min_cost = cost
                    best_thresh = thresh
                    best_cm = (tn, fp, fn, tp)
                    
        if best_cm:
            tn, fp, fn, tp = best_cm
            results_list.append({
                'FP_Cost': w_fp,
                'FN_Cost': w_fn,
                'Best_Thresh': best_thresh,
                'FP': fp,
                'FN': fn
            })

    df_spectrum = pd.DataFrame(results_list)
    
    # ---------------------------------------------------------
    # EXTRACT THE OPTIMAL "ZERO FP" RATIO & CONFUSION MATRIX
    # ---------------------------------------------------------
    df_safe = df_spectrum[df_spectrum['FP'] == 0]
    
    if not df_safe.empty:
        best_safe = df_safe.sort_values(by=['FN', 'FP_Cost'], ascending=[True, True]).iloc[0]
        
        fp_weight = int(best_safe['FP_Cost'])
        fn_weight = int(best_safe['FN_Cost'])
        min_fn = int(best_safe['FN'])
        optimal_thresh = best_safe['Best_Thresh']
        
        ratio = fp_weight / fn_weight if fn_weight > 0 else float('inf')
        
        optimal_ratios.append({
            'Threshold (Years)': yrs,
            'Optimal FP Cost': fp_weight,
            'Optimal FN Cost': fn_weight,
            'Equivalent Ratio': f"{ratio:.1f}:1",
            'Missed OD (FN)': min_fn
        })
        
        # --- CALCULATE KAPLAN-MEIER FOR THE FINAL TABLE ---
        y_pred_optimal = (y_probs >= optimal_thresh).astype(int)
        df_eval['Predicted_Class'] = y_pred_optimal
        df_eval['Event'] = np.where(df_eval['COD to site recode'] == 'Breast', 1, 0)
        T = df_eval['Survival months'] / 12.0
        E = df_eval['Event']
        kmf_agg = KaplanMeierFitter()
        mask_agg = (df_eval['Predicted_Class'] == 0)
        if sum(mask_agg) > 0:
            kmf_agg.fit(T[mask_agg], event_observed=E[mask_agg])
            kmf_aggressive_dict[yrs] = kmf_agg

        # ==========================================
        # SHOW ADVISOR'S FP vs FN TRADEOFF PLOT ON SCREEN
        # ==========================================
        print(f"\n[!] Displaying Tradeoff Plot for {yrs} Years...")
        df_plot = df_spectrum.sort_values('FP_Cost')

        plt.figure(figsize=(9, 6))
        
        plt.plot(df_plot['FP_Cost'], df_plot['FP'], label='False Positives (Lethal Error)', color='red', linewidth=3)
        plt.plot(df_plot['FP_Cost'], df_plot['FN'], label='False Negatives (Missed Overdiagnosis)', color='black', linestyle='--', linewidth=3)

        plt.title(f'{yrs}-Year Threshold: Error Trade-off vs. Unit Cost', fontsize=16, fontweight='bold')
        plt.xlabel('False Positive Unit Cost (Out of 100)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Misclassified Patients', fontsize=14, fontweight='bold')
        
        plt.xlim(0, 100)
        plt.axvline(x=fp_weight, color='blue', linestyle=':', linewidth=2, label=f'Max Safety Threshold (FP Cost = {fp_weight})')

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # THIS WILL POP THE GRAPH UP ON YOUR SCREEN INSTEAD OF SAVING IT
        plt.show() 
            
    else:
        print("\n[!] WARNING: Could not achieve 0 False Positives at this threshold, even at 100:0 cost.")
        optimal_ratios.append({
            'Threshold (Years)': yrs,
            'Optimal FP Cost': 'N/A',
            'Optimal FN Cost': 'N/A',
            'Equivalent Ratio': 'Safety Failed',
            'Missed OD (FN)': 'N/A'
        })

# ==========================================
# 5. FINAL SUMMARY REPORT
# ==========================================
print("\n" + "="*80)
print("   FINAL OPTIMIZATION SUMMARY ACROSS ALL THRESHOLDS   ")
print("="*80)
summary_df = pd.DataFrame(optimal_ratios)
print(summary_df.to_string(index=False))

# ==========================================
# 6. FINAL SURVIVAL TABLE DATA
# ==========================================
print("\n" + "="*80)
print("   DATA FOR LATEX SURVIVAL TABLE   ")
print("="*80)
for yrs, kmf in kmf_aggressive_dict.items():
    final_survival = kmf.survival_function_.iloc[-1, 0] * 100
    print(f"{yrs}-Year Aggressive Cohort Final Survival: {final_survival:.1f}%")
