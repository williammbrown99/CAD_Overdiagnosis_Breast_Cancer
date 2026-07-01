# Import pandas for tabular data manipulation
import pandas as pd
# Import numpy for fast mathematical operations and NaN handling
import numpy as np
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
filename = "INPUT/SEER_11760_alive_and_in_situ_included.csv"
#filename = "INPUT/SEER_556_alive_in_situ_refused.csv"

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
    'Histology recode - broad groupings', 'Site recode - rare tumors',
    'Marital status at diagnosis', 'Median household income inflation adj to 2023',
    'Primary Site', 'Laterality', 'Race recode (White, Black, Other)'
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
# 4. TEMPORAL EXPERT MODELS & METRICS CALCULATION
# ==========================================
thresholds_yrs = [5, 6, 7, 8, 9, 10]
all_metrics = []

# --- BOUNDARY CONSTRAINT ---
# Require the model to correctly identify at least 50% of the Aggressive tumors 
# to prevent the "predict everyone is indolent" trivial solution.
MIN_SPECIFICITY_FOR_INDOLENCE = 0.50 

def compute_metrics(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    return sensitivity, specificity, precision, f1

for yrs in thresholds_yrs:
    thresh_mos = yrs * 12 
    
    print("\n" + "="*80)
    print(f"   EVALUATING 3 STRATEGIES FOR {yrs}-YEAR THRESHOLD   ")
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

    # --- EXTRACT STABILITY METRICS (CROSS-VALIDATION VARIANCE) ---
    best_idx = random_search.best_index_
    cv_mean = random_search.cv_results_['mean_test_score'][best_idx]
    cv_std = random_search.cv_results_['std_test_score'][best_idx]
    print(f"[*] CV Stability - Mean F1: {cv_mean:.4f} | Std Dev: +/- {cv_std:.4f}")

    y_probs = best_xgb.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # RUN SLIDING SCALE OPIMIZATION
    # ---------------------------------------------------------
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
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            results_list.append({'FP_Cost': w_fp, 'FN_Cost': w_fn, 'Best_Thresh': best_thresh, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'Spec': spec})

    df_spectrum = pd.DataFrame(results_list)
    
    # ---------------------------------------------------------
    # STRATEGY 1: MAX CLINICAL SAFETY (FP = 0)
    # ---------------------------------------------------------
    df_safe = df_spectrum[df_spectrum['FP'] == 0]
    if not df_safe.empty:
        best_safe = df_safe.sort_values(by=['FN', 'FP_Cost'], ascending=[True, True]).iloc[0]
        tn, fp, fn, tp = int(best_safe['TN']), int(best_safe['FP']), int(best_safe['FN']), int(best_safe['TP'])
        sens, spec, prec, f1 = compute_metrics(tn, fp, fn, tp)
        all_metrics.append({
            'Threshold': f"{yrs} Years", 'Strategy': f"Max Clinical Safety ({int(best_safe['FP_Cost'])}:{int(best_safe['FN_Cost'])})",
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'Sens': sens, 'Spec': spec, 'Prec': prec, 'F1': f1
        })
    
    # ---------------------------------------------------------
    # STRATEGY 2: 1-TO-1 BALANCED COST (50:50)
    # ---------------------------------------------------------
    df_bal = df_spectrum[df_spectrum['FP_Cost'] == 50]
    if not df_bal.empty:
        best_bal = df_bal.iloc[0]
        tn, fp, fn, tp = int(best_bal['TN']), int(best_bal['FP']), int(best_bal['FN']), int(best_bal['TP'])
        sens, spec, prec, f1 = compute_metrics(tn, fp, fn, tp)
        all_metrics.append({
            'Threshold': f"{yrs} Years", 'Strategy': "1-to-1 Balanced Cost (50:50)",
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'Sens': sens, 'Spec': spec, 'Prec': prec, 'F1': f1
        })

    # ---------------------------------------------------------
    # STRATEGY 3: MAX INDOLENCE DETECTION (Bounded)
    # ---------------------------------------------------------
    # Filter to only include rows where Specificity >= MIN_SPECIFICITY_FOR_INDOLENCE
    df_bounded = df_spectrum[df_spectrum['Spec'] >= MIN_SPECIFICITY_FOR_INDOLENCE]
    
    if not df_bounded.empty:
        # Sort to find the lowest FN within the allowed Specificity boundary
        best_ind = df_bounded.sort_values(by=['FN', 'FN_Cost'], ascending=[True, True]).iloc[0]
        tn, fp, fn, tp = int(best_ind['TN']), int(best_ind['FP']), int(best_ind['FN']), int(best_ind['TP'])
        sens, spec, prec, f1 = compute_metrics(tn, fp, fn, tp)
        all_metrics.append({
            'Threshold': f"{yrs} Years", 'Strategy': f"Max Indolence (Bounded {int(best_ind['FP_Cost'])}:{int(best_ind['FN_Cost'])})",
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'Sens': sens, 'Spec': spec, 'Prec': prec, 'F1': f1
        })
    else:
        # Fallback if impossible
        best_ind = df_spectrum.sort_values(by=['FN']).iloc[0]
        tn, fp, fn, tp = int(best_ind['TN']), int(best_ind['FP']), int(best_ind['FN']), int(best_ind['TP'])
        sens, spec, prec, f1 = compute_metrics(tn, fp, fn, tp)
        all_metrics.append({
            'Threshold': f"{yrs} Years", 'Strategy': f"Max Indolence (Best Effort {int(best_ind['FP_Cost'])}:{int(best_ind['FN_Cost'])})",
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'Sens': sens, 'Spec': spec, 'Prec': prec, 'F1': f1
        })

# ==========================================
# 5. FINAL EXPORT FOR LATEX TABLE
# ==========================================
print("\n" + "="*110)
print("   FINAL METRICS TABLE FOR ALL STRATEGIES & THRESHOLDS   ")
print("="*110)
df_final_metrics = pd.DataFrame(all_metrics)

# Formatting to match the desired LaTeX table output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df_final_metrics['Sens'] = df_final_metrics['Sens'].apply(lambda x: f"{x:.2f}")
df_final_metrics['Spec'] = df_final_metrics['Spec'].apply(lambda x: f"{x:.2f}")
df_final_metrics['Prec'] = df_final_metrics['Prec'].apply(lambda x: f"{x:.2f}")
df_final_metrics['F1'] = df_final_metrics['F1'].apply(lambda x: f"{x:.2f}")

print(df_final_metrics.to_string(index=False))
