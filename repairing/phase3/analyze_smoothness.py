"""
分析几何平滑度的改善情况
基于已有的 phase3_surface_metrics.csv 数据
"""
import pandas as pd
from pathlib import Path

def main():
    base_dir = Path('/projappl/project_2016517/JunjieCheng/NeAR_fix_Public-Cardiac-CT-Dataset')
    
    print("="*80)
    print("几何平滑度分析 - 基于表面指标数据")
    print("="*80)
    
    # 从已有的surface metrics分析
    metrics_path = base_dir / 'repairing' / 'phase3' / 'phase3_surface_metrics.csv'
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        return
    
    df = pd.read_csv(metrics_path)
    print(f"从 {metrics_path} 加载数据")
    print(f"总记录数: {len(df)}")
    
    # 关注的类别: LAA(8), Coronary(9), PV(10)
    focus_classes = {8: 'LAA', 9: 'Coronary', 10: 'PV'}
    
    print("\n" + "="*80)
    print("修复后表面质量分析 (ASD = Average Surface Distance)")
    print("="*80)
    
    # 根据报告的等周比率数据
    print("\n### 项目报告中的等周比率 (Isoperimetric Ratio) 统计 ###")
    print("(数值越低 = 越平滑)")
    print("-"*60)
    print(f"{'类别':<12} | {'原始 Ratio':<12} | {'修复后 Ratio':<12} | {'改善':<10}")
    print("-"*60)
    print(f"{'Coronary':<12} | {'32.03':<12} | {'20.11':<12} | {'-37%':<10} ✅")
    print(f"{'LAA':<12} | {'14.45':<12} | {'11.96':<12} | {'-17%':<10}")
    print(f"{'PV':<12} | {'19.48':<12} | {'17.92':<12} | {'-8%':<10}")
    
    print("\n### 各类别的 ASD/HD95 统计 ###")
    for class_id, class_name in focus_classes.items():
        class_data = df[df['class_id'] == class_id]
        if len(class_data) > 0:
            print(f"\n【{class_name}】 ({len(class_data)} 样本)")
            print(f"  ASD: 均值={class_data['ASD'].mean():.2f}, 中位数={class_data['ASD'].median():.2f}, 最小={class_data['ASD'].min():.2f}")
            print(f"  HD95: 均值={class_data['HD95'].mean():.2f}, 中位数={class_data['HD95'].median():.2f}")
            
            # 找出表面质量最好的案例（ASD最低）
            best_cases = class_data.nsmallest(5, 'ASD')
            print(f"  表面质量最好的案例 (ASD最低):")
            for _, row in best_cases.iterrows():
                print(f"    Case {int(row['case_id'])}: ASD={row['ASD']:.2f}, HD95={row['HD95']:.2f}")
    
    # 结合拓扑数据找综合优秀案例
    print("\n" + "="*80)
    print("综合推荐: 平滑度良好且拓扑修复成功的案例")
    print("="*80)
    
    # 加载拓扑验证数据
    topo_path = base_dir / 'repairing' / 'phase3' / 'phase3_verification_full.csv'
    orig_topo_path = base_dir / 'repairing' / 'phase3' / 'original_topology_verification.csv'
    
    if topo_path.exists() and orig_topo_path.exists():
        df_topo = pd.read_csv(topo_path)
        df_orig = pd.read_csv(orig_topo_path)
        df_merged = pd.merge(df_topo, df_orig, on='case_id', suffixes=('_phase3', '_orig'))
        
        # 找出LAA修复成功的案例
        laa_fixed = df_merged[(df_merged['Conn_LAA_LA_orig'] < 1) & (df_merged['Conn_LAA_LA_phase3'] == 1)]
        
        # 取这些案例的冠脉ASD数据
        cor_data = df[df['class_id'] == 9]
        laa_data = df[df['class_id'] == 8]
        
        # 找出LAA修复成功且ASD较低的案例
        laa_fixed_ids = set(laa_fixed['case_id'].tolist())
        cor_good = cor_data[cor_data['case_id'].isin(laa_fixed_ids)].nsmallest(10, 'ASD')
        
        print("\n【推荐案例】LAA修复成功 + 冠脉表面质量好:")
        for _, row in cor_good.iterrows():
            case_id = int(row['case_id'])
            orig_laa = df_merged[df_merged['case_id'] == case_id]['Conn_LAA_LA_orig'].values[0]
            print(f"  Case {case_id}: LAA {orig_laa:.0%}->100%, Coronary ASD={row['ASD']:.2f}")
        
        # 找出原始LAA连接率最低但修复成功的
        worst_orig = laa_fixed.nsmallest(10, 'Conn_LAA_LA_orig')
        print("\n【推荐案例】原始LAA连接率最差但修复成功:")
        for _, row in worst_orig.iterrows():
            case_id = int(row['case_id'])
            orig_laa = row['Conn_LAA_LA_orig']
            # 获取该案例的LAA和Coronary ASD
            laa_asd = laa_data[laa_data['case_id'] == case_id]['ASD'].values
            cor_asd = cor_data[cor_data['case_id'] == case_id]['ASD'].values
            laa_asd_str = f"{laa_asd[0]:.2f}" if len(laa_asd) > 0 else "N/A"
            cor_asd_str = f"{cor_asd[0]:.2f}" if len(cor_asd) > 0 else "N/A"
            print(f"  Case {case_id}: LAA {orig_laa:.1%}->100%, LAA_ASD={laa_asd_str}, Cor_ASD={cor_asd_str}")

if __name__ == '__main__':
    main()
