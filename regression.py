"""
Logistic Regression Analysis for Gender Bias in Chinese-Portuguese AT
Chinese Challenge Set (EAMT 2026)

Requirements:
    pip install pandas numpy statsmodels openpyxl

Usage:
    python regression.py
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ── 1. Load data ──────────────────────────────────────────────────────────────

df = pd.read_excel('challenge_set.xlsx')

# System annotation column mapping
system_map = {
    'google\n[ann]':                              'Google',
    'deepl\n[ann]':                               'DeepL',
    'openai/gpt-5.2\n[ann]':                      'GPT5',
    'anthropic/claude-sonnet-4.6\n[ann]':         'Claude',
    'mistralai/ministral-14b-2512\n[ann]':        'Ministral',
    'qwen/qwen3.5-397b-a17b\n[ann]':              'Qwen',
    'meta-llama/llama-3.3-70b-instruct\n[ann]':   'LLaMA',
    'deepseek\n[ann]':                            'DeepSeek',
}

ann_cols = [c for c in df.columns if '[ann]' in c]


# ── 2. Reshape to long format ─────────────────────────────────────────────────

rows = []
for _, row in df.iterrows():
    tag = row['semantic_tag']

    # Parse semantic tag into experimental factors
    if tag == 'base_null':
        cue_type = 'baseline'
        target_gender = 'none'
        complexity = 'none'
        position = 'none'
        expected = None  # no determinate expected gender

    elif tag.startswith('exp_'):
        cue_type = 'explicit'
        target_gender = tag.split('_')[1]   # m or f
        complexity = 'none'
        position = 'none'
        expected = 'M' if target_gender == 'm' else 'F'

    else:
        # Coreference conditions: {ante|post}_{smp|cpx}_{m|f}
        parts = tag.split('_')
        cue_type      = 'pronoun'
        position      = parts[0]   # ante or post
        complexity    = parts[1]   # smp or cpx
        target_gender = parts[2]   # m or f
        expected      = 'M' if target_gender == 'm' else 'F'

    for col, sys_name in system_map.items():
        if col not in df.columns:
            continue
        ann = str(row[col]).strip().upper() if pd.notna(row[col]) else ''
        ann_clean = ann if ann in ['M', 'F'] else None   # exclude 2gen/unknown

        correct = None
        if expected is not None and ann_clean is not None:
            correct = 1 if ann_clean == expected else 0

        rows.append({
            'occupation':    row['occupation_en'],
            'bias_label':    row['bias_label'],
            'semantic_tag':  tag,
            'cue_type':      cue_type,
            'target_gender': target_gender,
            'complexity':    complexity,
            'position':      position,
            'system':        sys_name,
            'system_type':   'NMT' if sys_name in ['Google', 'DeepL'] else 'LLM',
            'annotation':    ann_clean,
            'expected':      expected,
            'correct':       correct,
        })

long_df = pd.DataFrame(rows)
print(f"Total rows:           {len(long_df)}")
print(f"Valid correct/wrong:  {long_df['correct'].notna().sum()}")


# ── 3. Filter to pronoun conditions only ──────────────────────────────────────
#
#   Excluded:
#     - base_null  : no determinate expected gender
#     - explicit   : ceiling accuracy, zero variance
#     - 2gen/unknown outputs : unclassifiable
#
pronoun_df = long_df[
    (long_df['cue_type'] == 'pronoun') &
    (long_df['correct'].notna())
].copy()

print(f"Pronoun condition rows (N): {len(pronoun_df)}")


# ── 4. Set reference categories ───────────────────────────────────────────────

for col, ref in [
    ('target_gender', 'm'),
    ('complexity',    'smp'),
    ('position',      'ante'),
    ('system_type',   'NMT'),
    ('bias_label',    'N'),
]:
    pronoun_df[col] = pd.Categorical(pronoun_df[col])
    cats = [ref] + [c for c in pronoun_df[col].cat.categories if c != ref]
    pronoun_df[col] = pronoun_df[col].cat.set_categories(cats)


# ── 5. Fit logistic regression (main effects) ─────────────────────────────────

model = smf.logit(
    'correct ~ C(target_gender, Treatment("m")) + '
    'C(complexity,    Treatment("smp"))          + '
    'C(position,      Treatment("ante"))         + '
    'C(system_type,   Treatment("NMT"))          + '
    'C(bias_label,    Treatment("N"))',
    data=pronoun_df
).fit(disp=0)


# ── 6. Print results table ────────────────────────────────────────────────────

params = model.params
conf   = model.conf_int()
pvals  = model.pvalues

name_map = {
    'Intercept':                                          'Intercept',
    'C(target_gender, Treatment("m"))[T.f]':             'Target gender: F (ref: M)',
    'C(complexity, Treatment("smp"))[T.cpx]':            'Complexity: complex (ref: simple)',
    'C(position, Treatment("ante"))[T.post]':            'Position: cataphoric (ref: anaphoric)',
    'C(system_type, Treatment("NMT"))[T.LLM]':           'System type: LLM (ref: NMT)',
    'C(bias_label, Treatment("N"))[T.M]':                'Stereotype: M-typed (ref: neutral)',
    'C(bias_label, Treatment("N"))[T.F]':                'Stereotype: F-typed (ref: neutral)',
}

def stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    if p < .10:  return '†'
    return ''

print('\n' + '='*75)
print('Logistic Regression — Predictors of Correct Gender Accuracy')
print(f'N = {int(model.nobs)}   Pseudo R² (McFadden) = {model.prsquared:.3f}'
      f'   AIC = {model.aic:.1f}')
print('='*75)
print(f"{'Predictor':<45} {'β':>7} {'OR':>6} {'95% CI':>16} {'p':>8} {'':>4}")
print('-'*75)

for raw, label in name_map.items():
    if raw == 'Intercept':
        continue
    b  = params[raw]
    OR = np.exp(b)
    lo = np.exp(conf.loc[raw, 0])
    hi = np.exp(conf.loc[raw, 1])
    p  = pvals[raw]
    print(f"{label:<45} {b:>+7.3f} {OR:>6.2f} [{lo:.2f}, {hi:.2f}]  {p:>8.4f} {stars(p):>4}")

print('-'*75)
print('*** p<.001  ** p<.01  * p<.05  † p<.10')


# ── 7. Supplementary: gender × complexity interaction test ───────────────────

model_int = smf.logit(
    'correct ~ C(target_gender, Treatment("m")) * C(complexity, Treatment("smp")) + '
    'C(position,    Treatment("ante"))  + '
    'C(system_type, Treatment("NMT"))   + '
    'C(bias_label,  Treatment("N"))',
    data=pronoun_df
).fit(disp=0)

lr_stat = -2 * (model.llf - model_int.llf)
lr_p    = stats.chi2.sf(lr_stat, df=1)
print(f'\nInteraction test (gender × complexity): χ²={lr_stat:.3f}, p={lr_p:.4f}')
