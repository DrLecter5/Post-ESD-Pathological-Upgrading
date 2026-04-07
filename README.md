# 病理升级在线预测平台 
 
- 不依赖 `scikit-learn`
- 不依赖 `torch`
- 不在云端重新拟合模型
- 直接使用锁定的 logistic-style 系数进行在线计算
- 更适合给审稿人直接打开链接

## 适合上传到 GitHub 仓库的文件
- `pathology_upgrade_streamlit_public_review_nosklearn.py`
- `requirements.txt`
- `Training_Cohort_Patients.csv`
- `Internal_Validation_Patients.csv`
- `External_Validation_Patients.csv`

## 部署到 Streamlit Cloud
1. 将主程序文件上传到 GitHub 仓库根目录  
2.  `requirements.txt`  
3. 在 Streamlit Community Cloud 选择主文件：
   `pathology_upgrade_streamlit_public_review_nosklearn.py`
4. Deploy

 
