# 阶段三实施方案：多类融合与解剖一致性修正 (Stage 3 Plan)

本方案旨在将 Stage 2 生成的 10 个单类 Mask (Per-class Ground Truth) 融合成一个无冲突、符合解剖学逻辑的最终多类分割结果 (Final Segmentation)。

## 1. 核心策略

整个过程分为三步：
1.  **全局冲突解决 (Priority Fusion)**：利用预定义的优先级列表，解决体素级别的类别冲突。
2.  **解剖一致性修正 (Anatomical Constraints)**：应用基于解剖学知识的规则，修复断连、穿插等几何错误。
3.  **最终清理 (Sanity Check)**：去除微小噪点。

## 2. 优先级列表 (Priority List)

为了确保关键结构不被覆盖，同时保证心肌壁的正确形态，我们采用以下优先级（从高到低）：

| Priority | Class ID | Name | Reason |
| :--- | :--- | :--- | :--- |
| **1** | 9 | **Coronary** | 细微结构，最易断裂，必须最优先保留。 |
| **2** | 10 | **PV** | 结构较散，需优先于心房保留。 |
| **3** | 8 | **LAA** | 左心耳，需优先于左心房。 |
| **4** | 3, 5, 2, 4 | **LV, RV, LA, RA** | **关键点**：腔室优先级高于心肌，以“挖空”实心心肌，还原心壁。 |
| **5** | 1 | **Myocardium** | 被腔室挖空后，剩余部分即为心肌壁。 |
| **6** | 6 | **Aorta** | 大血管。 |
| **7** | 7 | **PA** | 大血管。 |
| **8** | 0 | **Background** | |

*注：腔室内部 (LV/RV/LA/RA) 之间若有重叠，通常由网络预测置信度决定，但在 Mask 融合阶段，可按 ID 顺序或视为互斥（通常不会重叠太多）。*

## 3. 解剖一致性规则 (Anatomy Rules)

在优先级融合之后，执行以下规则修正几何形态：

### Rule 1: 腔室包络规则 (Chamber Enclosure)
*   **目标**：确保心肌 (Myo) 紧密包裹 LV/RV/LA/RA，且腔室不泄露到 Myo 外界。
*   **实现**：
    *   以 Myo 为基础做轻微膨胀 (Dilate) 作为“安全边界”。
    *   任何超出此边界的腔室体素（且不属于 Ao/PA 等其他结构）视为“泄漏”，强制改为 Background 或 Myo。

### Rule 2: 腔室去噪 (Chamber Denoising)
*   **目标**：去除腔室 Mask 边缘的毛刺或非生理性突起。
*   **实现**：
    *   对每个腔室做连通分量分析。
    *   若存在次大分量且距离主分量极近 (<3 voxels)，则合并（视为连接处的断裂）。
    *   否则删除次大分量。

### Rule 3: PV 连接性修正 (PV-LA Connection)
*   **目标**：肺静脉 (PV) 必须连接到左心房 (LA)。
*   **实现**：
    *   计算 PV 各分量到 LA 的最短距离。
    *   若距离 > 阈值 (e.g., 5 voxels) → 视为孤立噪点，**删除**。
    *   若距离在 1~5 voxels 之间 → 做定向膨胀 (Dilate) 强制连接到 LA。

### Rule 4: LAA 连接性修正 (LAA-LA Connection)
*   **目标**：左心耳 (LAA) 必须连接到左心房 (LA)。
*   **实现**：同 Rule 3。

### Rule 5: 冠脉依附规则 (Coronary Attachment)
*   **目标**：冠脉 (Coronary) 必须依附于心肌 (Myo) 或主动脉 (Aorta) 表面。
*   **实现**：
    *   计算 Coronary 各分量到 (Myo ∪ Aorta) 的最短距离。
    *   若整个分量都远离表面 (> 5 voxels) → 视为漂浮噪点，**删除**。
    *   保留依附的分量。

### Rule 6: 大血管互斥 (Vessel Exclusion)
*   **目标**：Aorta 和 PA 不应穿透心肌进入腔室。
*   **实现**：
    *   若 Aorta/PA 与 Myo 内表面（Eroded Myo）有重叠，强制归类为 Myo（修复穿插）。

### Rule 7: 全局清理 (Global Sanity)
*   **目标**：去除极小的孤立噪点。
*   **实现**：
    *   对所有类，删除体积 < 阈值 (e.g., 大结构 500, 小结构 50) 的独立分量。

## 4. 实施步骤

1.  **数据准备**：读取 Stage 2 输出的 10 个 `.npy` 文件。
2.  **执行融合**：运行 `stage3.py`，依次应用上述逻辑。
3.  **输出结果**：生成 `final_segmentation.nii.gz` (11类) 和 `final_segmentation.npy`。
4.  **验证**：使用 `stage3_run_case.py` 可视化对比修正前后的差异。

## 5. 文件路径
*   **代码**: `/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/stage3/`
*   **输入**: `.../repairing/stage2/classX_..._processed/*.npy`
*   **输出**: `.../repairing/stage3/output/`
