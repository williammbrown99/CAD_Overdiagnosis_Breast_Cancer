import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA & BASE CLEANING
# ==========================================
# UPDATED: Pointing to the final alive-included dataset
filename = "INPUT/SEER_11760_alive_and_in_situ_included.csv"
print(f"Loading {filename} for Baseline Characteristics...\n")
df = pd.read_csv(filename, low_memory=False)

df['Survival months'] = pd.to_numeric(df['Survival months'], errors='coerce')
df['Year of diagnosis'] = pd.to_numeric(df['Year of diagnosis'], errors='coerce')

# VITAL FILTER 1 & 2: Active survival > 0 and exclude post-mortem diagnoses
df = df[df['Survival months'] > 0] 
if 'Type of Reporting Source' in df.columns:
    autopsy_mask = df['Type of Reporting Source'].astype(str).str.contains('autopsy|death certificate', case=False, na=False)
    df = df[~autopsy_mask]

df = df[(df['Year of diagnosis'] >= 1990) & (df['Year of diagnosis'] <= 2016)].copy()

df_base = df[(df['Marital status at diagnosis'] != 'Unknown') & 
             (~df['Median household income inflation adj to 2023'].str.contains('Unknown', na=False, case=False))].copy()

# ==========================================
# 2. FEATURE ENGINEERING (With the NaN Fix)
# ==========================================
size_cols = ['Tumor Size Summary (2016+)', 'CS tumor size (2004-2015)', 'EOD 10 - size (1988-2003)', 'EOD 4 - size (1983-1987)']
def clean_size(val):
    try:
        v = float(val)
        if v == 990: return 0.5
        return v if 0 <= v <= 400 else np.nan
    except: return np.nan

# THE FIX: Clean all columns FIRST to convert text strings to true NaNs
for col in size_cols:
    if col in df_base.columns:
        df_base[col] = df_base[col].apply(clean_size)
# THEN backfill to capture historical measurements perfectly
df_base['Master_Tumor_Size'] = df_base[size_cols].bfill(axis=1).iloc[:, 0]

grade_cols = ['Grade Recode (thru 2017)', 'Grade Clinical (2018+)', 'Grade Pathological (2018+)']
def clean_grade(val):
    v = str(val).lower()
    if any(x in v for x in ['3', '4', 'poor', 'undiff', 'iii', 'iv']): return 'Grade 3/4 (Poor/Undiff)'
    if any(x in v for x in ['2', 'mod', 'ii']): return 'Grade 2 (Moderate)'
    if any(x in v for x in ['1', 'well', 'grade i']) and 'ii' not in v: return 'Grade 1 (Well)'
    return np.nan
df_base['Master_Grade'] = np.nan
for col in grade_cols:
    if col in df_base.columns: df_base['Master_Grade'] = df_base['Master_Grade'].fillna(df_base[col].apply(clean_grade))

stage_cols = ['Summary stage 2000 (1998-2017)', 'SEER historic stage A (1973-2015)', 'Combined Summary Stage with Expanded Regional Codes (2004+)']
def clean_stage(val):
    v = str(val).lower()
    if 'distant' in v: return '3: Distant'
    if 'regional' in v: return '2: Regional'
    if 'localized' in v: return '1: Localized'
    if 'in situ' in v: return '0: In Situ'
    return np.nan
df_base['Master_Stage'] = np.nan
for col in stage_cols:
    if col in df_base.columns: df_base['Master_Stage'] = df_base['Master_Stage'].fillna(df_base[col].apply(clean_stage))

age_candidates = ['Age recode with single ages and 90+', 'Age recode with single ages and 85+', 'Age recode with <1 year olds']
actual_age_col = next((c for c in age_candidates if c in df_base.columns), None)
df_base['Age_Numeric'] = df_base[actual_age_col].apply(lambda x: int(str(x).split()[0]) if pd.notnull(x) else np.nan)

df_base['Has_Previous_History'] = df_base['Sequence number'].apply(
    lambda x: 'Yes' if any(word in str(x) for word in ['2nd', '3rd', '4th', '5th']) else 'No'
)

for col in ['ER Status Recode Breast Cancer (1990+)', 'PR Status Recode Breast Cancer (1990+)']:
    if col in df_base.columns:
        df_base[col] = df_base[col].replace(['Borderline/Unknown', 'Recode not available', 'Unknown', 'Blank(s)'], np.nan)

# ==========================================
# 3. AUTOMATED LOOP FOR TABLE 1 GENERATION
# ==========================================
# Evaluate the full 5-to-10 year continuum
thresholds_yrs = [5, 6, 7, 8, 9, 10]

for yrs in thresholds_yrs:
    thresh_mos = yrs * 12

    # The Aggressive class (< thresh_mos) now explicitly scales cumulatively with the Overdiagnosed class (>= thresh_mos, != Breast)
    cond_1 = (df_base['COD to site recode'] != 'Breast') & (df_base['Survival months'] >= thresh_mos)
    cond_0 = (df_base['COD to site recode'] == 'Breast') & (df_base['Survival months'] < thresh_mos)

    df_run = df_base[cond_1 | cond_0].copy()
    df_run['Cohort'] = np.where(
        (df_run['COD to site recode'] != 'Breast') & (df_run['Survival months'] >= thresh_mos), 
        'Indolent', 'Aggressive'
    )

    cohort_counts = df_run['Cohort'].value_counts()
    
    print("\n\n" + "="*95)
    print(f"{f'TABLE 1 - BASELINE COHORT CHARACTERISTICS ({yrs}-YEAR SCALED OVERDIAGNOSIS)':^95}")
    print("="*95)
    print(f"{'Feature':<35} | {'Indolent (n=' + str(cohort_counts.get('Indolent', 0)) + ')':<22} | {'Aggressive (n=' + str(cohort_counts.get('Aggressive', 0)) + ')':<22} | {'p-value':<10}")
    print("-" * 95)

    # Function for Continuous Variables
    def print_continuous(feature_name, col_name):
        ind_data = df_run[df_run['Cohort'] == 'Indolent'][col_name].dropna()
        agg_data = df_run[df_run['Cohort'] == 'Aggressive'][col_name].dropna()
        
        if len(ind_data) > 0 and len(agg_data) > 0:
            ind_median, ind_q1, ind_q3 = ind_data.median(), ind_data.quantile(0.25), ind_data.quantile(0.75)
            agg_median, agg_q1, agg_q3 = agg_data.median(), agg_data.quantile(0.25), agg_data.quantile(0.75)
            stat, p_val = mannwhitneyu(ind_data, agg_data, alternative='two-sided')
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        else:
            ind_median, ind_q1, ind_q3, agg_median, agg_q1, agg_q3 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            p_str = "NA"
        
        ind_str = f"{ind_median:.1f} ({ind_q1:.1f}-{ind_q3:.1f})"
        agg_str = f"{agg_median:.1f} ({agg_q1:.1f}-{agg_q3:.1f})"
        
        print(f"{feature_name:<35} | {ind_str:<22} | {agg_str:<22} | {p_str:<10}")

    # Function for Categorical Variables
    def print_categorical(feature_name, col_name):
        print(f"{feature_name}")
        crosstab = pd.crosstab(df_run[col_name], df_run['Cohort'])
        
        try:
            chi2, p_val, dof, expected = chi2_contingency(crosstab)
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        except:
            p_str = "NA"
            
        printed_p_val = False
        
        for category in crosstab.index:
            ind_count = crosstab.loc[category, 'Indolent'] if 'Indolent' in crosstab.columns else 0
            agg_count = crosstab.loc[category, 'Aggressive'] if 'Aggressive' in crosstab.columns else 0
            
            ind_pct = (ind_count / cohort_counts.get('Indolent', 1)) * 100
            agg_pct = (agg_count / cohort_counts.get('Aggressive', 1)) * 100
            
            ind_str = f"{ind_count} ({ind_pct:.0f}%)"
            agg_str = f"{agg_count} ({agg_pct:.0f}%)"
            
            display_p = p_str if not printed_p_val else ""
            printed_p_val = True
            
            print(f"  {category:<33} | {ind_str:<22} | {agg_str:<22} | {display_p:<10}")

    # --- PRINT THE TABLE FOR THE CURRENT THRESHOLD ---
    print_continuous("Age at Diagnosis (years)", "Age_Numeric")
    print_continuous("Tumor Size (mm)", "Master_Tumor_Size")
    print_categorical("Tumor Grade", "Master_Grade")
    print_categorical("SEER Summary Stage", "Master_Stage")

    if 'ER Status Recode Breast Cancer (1990+)' in df_run.columns:
        print_categorical("ER Status", "ER Status Recode Breast Cancer (1990+)")
    if 'PR Status Recode Breast Cancer (1990+)' in df_run.columns:
        print_categorical("PR Status", "PR Status Recode Breast Cancer (1990+)")
        
    print_categorical("Previous Cancer History", "Has_Previous_History")
    
    # NEW: Print the newly added features for your demographic/baseline table
    if 'Race recode (White, Black, Other)' in df_run.columns:
        print_categorical("Race", "Race recode (White, Black, Other)")
    if 'Primary Site' in df_run.columns:
        print_categorical("Primary Site", "Primary Site")
    if 'Laterality' in df_run.columns:
        print_categorical("Laterality", "Laterality")
        
    print("-" * 95)
