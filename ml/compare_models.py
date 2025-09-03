import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_train, X_test, y_train, y_test, threshold=0.5):
    """Evaluate model performance"""
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'avg_precision': average_precision_score(y_test, y_proba)
    }

def main():
    print("🔍 Comparing different algorithms for fraud detection...")
    
    # Load data
    df = pd.read_csv('creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Original training set: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
    print(f"After SMOTE: {X_train_smote.shape}, Fraud rate: {y_train_smote.mean():.4f}")
    
    # Define models to compare
    models = {
        'Current XGBoost': XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            random_state=42, n_jobs=-1
        ),
        'Improved XGBoost': XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            reg_alpha=1.0, reg_lambda=2.0, subsample=0.8,
            colsample_bytree=0.8, eval_metric='aucpr',
            random_state=42, n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            penalty='l1', C=0.1, class_weight='balanced',
            solver='liblinear', random_state=42, max_iter=1000
        )
    }
    
    results = []
    
    print("\n" + "="*80)
    print("🏆 MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<10} {'F1':<10} {'PR-AUC':<10}")
    print("-"*80)
    
    for name, model in models.items():
        try:
            metrics = evaluate_model(model, X_train_smote, X_test_scaled, y_train_smote, y_test)
            results.append({
                'model': name,
                **metrics
            })
            
            print(f"{name:<20} {metrics['precision']:<12.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} {metrics['avg_precision']:<10.4f}")
                  
        except Exception as e:
            print(f"{name:<20} ERROR: {str(e)}")
    
    # Find best models
    results_df = pd.DataFrame(results)
    best_precision = results_df.loc[results_df['precision'].idxmax()]
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    best_pr_auc = results_df.loc[results_df['avg_precision'].idxmax()]
    
    print("\n" + "="*80)
    print("🥇 BEST PERFORMERS")
    print("="*80)
    print(f"Best Precision: {best_precision['model']} ({best_precision['precision']:.4f})")
    print(f"Best F1 Score:  {best_f1['model']} ({best_f1['f1']:.4f})")
    print(f"Best PR-AUC:    {best_pr_auc['model']} ({best_pr_auc['avg_precision']:.4f})")
    
    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS FOR PRECISION IMPROVEMENT")
    print("="*80)
    
    current_precision = results_df[results_df['model'] == 'Current XGBoost']['precision'].iloc[0]
    best_precision_score = results_df['precision'].max()
    improvement = ((best_precision_score / current_precision) - 1) * 100
    
    print(f"1. Switch to {best_precision['model']} for {improvement:+.1f}% precision boost")
    print(f"2. Use ensemble of top 3 models for even better results")
    print(f"3. Optimize threshold using precision-recall curve")
    print(f"4. Consider cost-sensitive learning with custom loss functions")
    print(f"5. Feature engineering: interaction terms, polynomial features")

if __name__ == "__main__":
    main()