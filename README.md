更新说明（v2）

这版会优先读取以下文件来重建训练/验证队列：
- Training_Cohort_Patients.csv
- Internal_Validation_Patients.csv
- External_Validation_Patients.csv

核心逻辑：
1. 把原始变量来源表（训练集.xlsx + 验证集.xlsx）先合并成总池
2. 再按新的 cohort 名单重新划分
3. 网站中的模型训练只用 Training_Cohort_Patients.csv 对应患者
4. Internal / External 名单仅用于页面说明，不参与重新拟合

因此，如果你已经把一部分病人转为外部验证，建议使用这版。

