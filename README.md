# MT-ADSCCA
The source code is an implementation of our method described in the paper "Adaptive Deep Sparse Canonical Correlation-based Multi-omics Gene Selection for Cancer Survival Analysis". 


--Dependence

Before running this code, you need install python. In our experiment, python 3.6 or more advanced version is tested. This code is tested in WIN7/win10 64 Bit. It should be able to run in other Linux or Windows systems.

--How to run 
There are three folders (BRCA, GBMLGG, and KIPAN). Here, we just take BRCA data as an example to illustrate this method. Same to other two datasets. 

step1. In folder: /MT-ADSCCA/BRCA/, first,using python, run brca_methylation_mRNA_MT_ADSCCA_method.py. If the code runs successfully, the extracted integrative mRNA features and methylation features will be obtained using MT-ADSCCA method. we can obtain a feature: BRCA_data_mm_name_dscca_p001_8-9_20210910.csv as the input for LSTM network.

step2. Run brca_MT_ADSCCA_survival-analysis_using_biLSTM.py.

