# Telcom Churn 
It is a project of combining **supervised models** -  
        1.KNN
        2.SVM 
        3.Logistic Regression 
        4.Naive-Bayes 
        5.Neural net 
    
**unsupervised models** - 
        1.PCA 
***The Flowchar of the project is*** 
               Telcom Churn
                     ↓
       "Which customers will leave?"
                     ↓
                GET DATA(telco.csv)
                     ↓
              UNDERSTAND DATA
                     ↓
                   EDA
                     ↓
            PREPARE THE DATA
                     ↓
              SPLIT THE DATA
                     ↓
           ┌─────────┴─────────┐
           ↓                   ↓
     SUPERVISED            UNSUPERVISED
           ↓                   ↓
   Churn Prediction           PCA
           ↓                   ↓
 Logistic/KNN/SVM/       Customer segments
 Naive Bayes/NN               ↓
           ↓              Analyze clusters
      Evaluation               ↓
           ↓            Churn by cluster
           └─────────┬─────────┘
                     ↓
               FINAL RESULTS