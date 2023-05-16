
# Glass Transition Temperature Machine Learning Models Repo

## Abstract

The accurate prediction of the glass transition temperature (Tg) holds significant importance in materials science, providing valuable insights into the behaviour and performance of polymer materials. Data-driven approaches utilizing machine learning techniques have gained prominence over traditional experimental methods for Tg prediction. This study performs a comparative evaluation of two ensemble machine learning algorithms, Random Forest and XGBoost, to assess their effectiveness in predicting Tg. Four different feature representations, namely ECFP_1024, ECFP_2048, 2D_descriptors, and 3D_descriptors, are also investigated for their predictive capabilities. The study employs rigorous experimentation and analysis to evaluate the performance of the machine learning algorithms and feature representations in Tg prediction. Evaluation metrics such as coefficient correlation, R2, mean absolute error (MAE) amd root mean squared error (RMSE) were used to evaluate the model. It was identified that 3D descriptors were the worst feature representation among the four. XGBoost and Random Forest performed well but on different representations respectively. 


## Usage

The repository for the glass transition temperature machine learning models. 

ECFP 1024, 2048 were are generated in ```Both ECFP.py```

2D and 3D descriptors are in their respective python files. 

Ensure that you change the directory link when using your own dataset. 
