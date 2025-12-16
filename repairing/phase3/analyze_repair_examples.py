"""
分析修复结果，找出成功、无变化和失败的典型案例
"""
import pandas as pd
import numpy as np

# Load data
df_phase3 = pd.read_csv('phase3_verification_full.csv')
df_orig = pd.read_csv('original_topology_verification.csv')

# Merge the dataframes
df = pd.merge(df_phase3, df_orig, on='case_id', suffixes=('_phase3', '_orig'))

# 确保连通性列是数值类型
for col in ['Conn_PV_LA_orig', 'Conn_LAA_LA_orig', 'Conn_Cor_MyoAo_orig',
            'Conn_PV_LA_phase3', 'Conn_LAA_LA_phase3', 'Conn_Cor_MyoAo_phase3']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("="*80)
print("修复结果分析 - 找寻典型案例")
print("="*80)

# --- 1. 冠脉连通性修复 (CC_Coronary) ---
print("\n### 1. 冠脉 (Coronary) 连通性修复分析 ###")
coronary_cc = df_phase3['CC_Coronary']
print(f"修复后冠脉CC分布:")
print(f"  CC=1: {sum(coronary_cc == 1)} 样本 ({sum(coronary_cc == 1)/len(coronary_cc)*100:.1f}%)")
print(f"  CC=2: {sum(coronary_cc == 2)} 样本 ({sum(coronary_cc == 2)/len(coronary_cc)*100:.1f}%)")
print(f"  CC>2: {sum(coronary_cc > 2)} 样本 ({sum(coronary_cc > 2)/len(coronary_cc)*100:.1f}%)")

# --- 2. LAA-LA 连接修复 ---
print("\n### 2. 左心耳-左心房 (LAA-LA) 连接修复分析 ###")
laa_orig = df['Conn_LAA_LA_orig']
laa_phase3 = df['Conn_LAA_LA_phase3']

laa_success = (laa_orig < 1) & (laa_phase3 == 1)
laa_nochange = (laa_orig == 1) & (laa_phase3 == 1)

print(f"原始数据 LAA-LA 连接率 < 100%: {sum(laa_orig < 1)} 样本")
print(f"修复成功 (从断开修复到连接): {sum(laa_success)} 样本")
print(f"本身就是好的: {sum(laa_nochange)} 样本")
print(f"修复后 100% 连接: {sum(laa_phase3 == 1)} 样本")

# Success examples for LAA
laa_success_df = df[laa_success].copy()
laa_success_df['improvement'] = laa_success_df['Conn_LAA_LA_phase3'] - laa_success_df['Conn_LAA_LA_orig']
laa_success_df = laa_success_df.sort_values('Conn_LAA_LA_orig')
print(f"\nLAA修复成功的典型案例 (原始连接率最低的):")
for _, row in laa_success_df.head(5).iterrows():
    print(f"  Case {int(row['case_id'])}: 原始 {row['Conn_LAA_LA_orig']:.2%} -> 修复后 {row['Conn_LAA_LA_phase3']:.2%}")

# --- 3. PV-LA 连接修复 ---
print("\n### 3. 肺静脉-左心房 (PV-LA) 连接修复分析 ###")
pv_orig = df['Conn_PV_LA_orig']
pv_phase3 = df['Conn_PV_LA_phase3']

pv_success = (pv_orig < 1) & (pv_phase3 == 1)
pv_nochange = (pv_orig == 1) & (pv_phase3 == 1)

print(f"原始数据 PV-LA 连接率 < 100%: {sum(pv_orig < 1)} 样本")
print(f"修复成功 (从断开修复到连接): {sum(pv_success)} 样本")
print(f"本身就是好的: {sum(pv_nochange)} 样本")

pv_success_df = df[pv_success].copy()
print(f"\nPV修复成功的典型案例:")
for _, row in pv_success_df.head(5).iterrows():
    print(f"  Case {int(row['case_id'])}: 原始 {row['Conn_PV_LA_orig']:.2%} -> 修复后 {row['Conn_PV_LA_phase3']:.2%}")

# --- 4. 冠脉依附性 ---
print("\n### 4. 冠脉依附性 (Coronary Attachment) 分析 ###")
cor_orig = df['Conn_Cor_MyoAo_orig']
cor_phase3 = df['Conn_Cor_MyoAo_phase3']

cor_success = (cor_orig < 1) & (cor_phase3 == 1)
cor_nochange = (cor_orig == 1) & (cor_phase3 == 1)
cor_failure = (cor_phase3 < 1) & (cor_orig == 1)  # Was good, now worse
cor_still_bad = (cor_phase3 < 1) & (cor_orig < 1)  # Still not fully attached

print(f"原始数据冠脉依附率 < 100%: {sum(cor_orig < 1)} 样本")
print(f"修复成功: {sum(cor_success)} 样本")
print(f"本身就是好的: {sum(cor_nochange)} 样本")
print(f"修复后依附率下降 (潜在失败): {sum(cor_failure)} 样本")
print(f"仍未完全依附: {sum(cor_still_bad)} 样本")
print(f"修复后 100% 依附: {sum(cor_phase3 == 1)} 样本 ({sum(cor_phase3 == 1)/len(cor_phase3)*100:.1f}%)")

# Success examples
cor_success_df = df[cor_success].copy()
if len(cor_success_df) > 0:
    print(f"\n冠脉依附修复成功的典型案例:")
    for _, row in cor_success_df.head(5).iterrows():
        print(f"  Case {int(row['case_id'])}: 原始 {row['Conn_Cor_MyoAo_orig']:.2%} -> 修复后 {row['Conn_Cor_MyoAo_phase3']:.2%}")

# Failure examples
cor_failure_df = df[cor_failure].copy()
if len(cor_failure_df) > 0:
    print(f"\n冠脉依附下降的案例 (需检查):")
    for _, row in cor_failure_df.head(5).iterrows():
        print(f"  Case {int(row['case_id'])}: 原始 {row['Conn_Cor_MyoAo_orig']:.2%} -> 修复后 {row['Conn_Cor_MyoAo_phase3']:.2%}")

print("\n" + "="*80)
print("总体修复统计")
print("="*80)

# Overall success rate
total = len(df)
print(f"\n总样本数: {total}")

# Count samples where at least one metric improved
any_improved = ((laa_orig < 1) & (laa_phase3 > laa_orig)) | \
               ((pv_orig < 1) & (pv_phase3 > pv_orig)) | \
               ((cor_orig < 1) & (cor_phase3 > cor_orig))

# Count samples that were already perfect
already_perfect = (laa_orig == 1) & (pv_orig == 1) & (cor_orig == 1)

print(f"至少一项拓扑指标得到改善: {sum(any_improved)} 样本 ({sum(any_improved)/total*100:.1f}%)")
print(f"原本所有拓扑就是完美的: {sum(already_perfect)} 样本 ({sum(already_perfect)/total*100:.1f}%)")

# Print some comprehensive examples
print("\n" + "="*80)
print("综合案例推荐 (可用于可视化)")
print("="*80)

# Best success case - most metrics improved
df['improvement_score'] = (laa_phase3 - laa_orig) + (pv_phase3 - pv_orig) + (cor_phase3 - cor_orig)
best_improvements = df.nlargest(10, 'improvement_score')
print("\n【修复成功案例】(改善幅度最大的):")
for _, row in best_improvements.iterrows():
    print(f"  Case {int(row['case_id'])}: LAA {row['Conn_LAA_LA_orig']:.2%}->{row['Conn_LAA_LA_phase3']:.2%}, "
          f"PV {row['Conn_PV_LA_orig']:.2%}->{row['Conn_PV_LA_phase3']:.2%}, "
          f"Cor {row['Conn_Cor_MyoAo_orig']:.2%}->{row['Conn_Cor_MyoAo_phase3']:.2%}, "
          f"Dice_LV={row['Dice_LV']:.3f}")

# No change case - was already perfect
no_change_cases = df[already_perfect].head(10)
print("\n【无变化案例】(原本就完美的):")
for _, row in no_change_cases.iterrows():
    print(f"  Case {int(row['case_id'])}: LAA {row['Conn_LAA_LA_orig']:.2%}->{row['Conn_LAA_LA_phase3']:.2%}, "
          f"PV {row['Conn_PV_LA_orig']:.2%}->{row['Conn_PV_LA_phase3']:.2%}, "
          f"Cor {row['Conn_Cor_MyoAo_orig']:.2%}->{row['Conn_Cor_MyoAo_phase3']:.2%}, "
          f"Dice_LV={row['Dice_LV']:.3f}")

# Potential failure - metrics degraded
failure_cases = df[(cor_phase3 < cor_orig) | ((laa_phase3 < 1) & (laa_orig == 1))]
print(f"\n【潜在失败案例】(某些指标下降的，共 {len(failure_cases)} 个):")
for _, row in failure_cases.head(10).iterrows():
    print(f"  Case {int(row['case_id'])}: LAA {row['Conn_LAA_LA_orig']:.2%}->{row['Conn_LAA_LA_phase3']:.2%}, "
          f"PV {row['Conn_PV_LA_orig']:.2%}->{row['Conn_PV_LA_phase3']:.2%}, "
          f"Cor {row['Conn_Cor_MyoAo_orig']:.2%}->{row['Conn_Cor_MyoAo_phase3']:.2%}, "
          f"Dice_Coronary={row['Dice_Coronary']:.3f}")

# Special case - Case 75 mentioned in the report
print("\n" + "="*80)
print("特别案例: Sample 75 (报告中提到的冠脉丢失案例)")
print("="*80)
case_75 = df[df['case_id'] == 75]
if len(case_75) > 0:
    row = case_75.iloc[0]
    print(f"  Case 75 在修复数据中")
    print(f"  Dice_Coronary: {row['Dice_Coronary']:.3f}")
    print(f"  CC_Coronary: {row['CC_Coronary']}")
else:
    print("  Case 75 不在数据中")

# Look for cases with very low coronary Dice (potential complete loss)
low_coronary = df[df['Dice_Coronary'] < 0.1]
print(f"\n冠脉Dice极低(<0.1)的案例 (可能冠脉丢失): {len(low_coronary)} 个")
for _, row in low_coronary.head(5).iterrows():
    print(f"  Case {int(row['case_id'])}: Dice_Coronary={row['Dice_Coronary']:.4f}, CC_Coronary={row['CC_Coronary']}")
