import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal threshold to maximize precision while maintaining reasonable recall"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    if metric == 'f1':
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
    elif metric == 'precision':
        # Find threshold that gives precision > 0.8 with highest recall
        high_precision_mask = precisions >= 0.8
        if np.any(high_precision_mask):
            optimal_idx = np.argmax(recalls[high_precision_mask])
            optimal_idx = np.where(high_precision_mask)[0][optimal_idx]
        else:
            optimal_idx = np.argmax(precisions)
    else:  # balanced
        # Maximize precision * recall (geometric mean)
        balanced_scores = np.sqrt(precisions * recalls)
        optimal_idx = np.argmax(balanced_scores)
    
    return thresholds[optimal_idx], precisions[optimal_idx], recalls[optimal_idx]

def create_ensemble_model(X_train, y_train, class_weights):
    """Create ensemble of different algorithms optimized for precision"""
    
    # XGBoost with precision focus
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=class_weights[1]/class_weights[0],
        eval_metric="aucpr",  # Optimize for precision-recall AUC
        random_state=42,
        n_jobs=-1
    )
    
    # LightGBM with precision focus
    lgb = LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        class_weight='balanced',
        metric='auc',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Random Forest with precision focus
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Logistic Regression with L1 regularization
    lr = LogisticRegression(
        penalty='l1',
        C=0.1,
        class_weight='balanced',
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    
    # Ensemble with soft voting
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('lgb', lgb), 
            ('rf', rf),
            ('lr', lr)
        ],
        voting='soft'
    )
    
    return ensemble

def main(args):
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("🚀 Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    target = "Class"
    features = [c for c in df.columns if c != target]

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], 
        test_size=0.2, 
        stratify=df[target], 
        random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"📊 Class weights: {class_weight_dict}")
    print(f"📊 Original fraud rate: {y_train.mean():.4f}")

    # Advanced resampling strategy
    if args.resampling == 'smote':
        resampler = SMOTE(random_state=42, sampling_strategy=args.sampling_strategy)
    elif args.resampling == 'adasyn':
        resampler = ADASYN(random_state=42, sampling_strategy=args.sampling_strategy)
    elif args.resampling == 'smoteenn':
        resampler = SMOTEENN(random_state=42, sampling_strategy=args.sampling_strategy)
    else:
        resampler = None

    if resampler:
        print(f"🔄 Applying {args.resampling} resampling...")
        X_resampled, y_resampled = resampler.fit_resample(X_train_scaled, y_train)
        print(f"📊 After resampling fraud rate: {y_resampled.mean():.4f}")
    else:
        X_resampled, y_resampled = X_train_scaled, y_train

    # Train models
    print("🤖 Training ensemble model...")
    
    if args.model_type == 'ensemble':
        model = create_ensemble_model(X_resampled, y_resampled, class_weights)
    elif args.model_type == 'xgb_tuned':
        # Hyperparameter tuning for XGBoost
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.03, 0.05, 0.08],
            'n_estimators': [300, 500],
            'reg_alpha': [0.5, 1.0, 2.0]
        }
        
        xgb_base = XGBClassifier(
            scale_pos_weight=class_weights[1]/class_weights[0],
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model = GridSearchCV(
            xgb_base, param_grid, 
            cv=cv, scoring='average_precision', 
            n_jobs=-1, verbose=1
        )
    else:  # default improved XGBoost
        model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=class_weights[1]/class_weights[0],
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_resampled, y_resampled)

    # Predictions and optimal threshold
    print("📈 Finding optimal threshold...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    optimal_threshold, opt_precision, opt_recall = find_optimal_threshold(
        y_test, y_proba, metric=args.threshold_metric
    )
    
    print(f"🎯 Optimal threshold: {optimal_threshold:.4f}")
    print(f"🎯 Expected precision: {opt_precision:.4f}, recall: {opt_recall:.4f}")

    # Final predictions with optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    y_pred_default = (y_proba >= 0.5).astype(int)

    # Metrics
    ap = average_precision_score(y_test, y_proba)
    
    # Default threshold metrics
    p_def, r_def, f1_def, _ = precision_recall_fscore_support(y_test, y_pred_default, average="binary")
    
    # Optimal threshold metrics  
    p_opt, r_opt, f1_opt, _ = precision_recall_fscore_support(y_test, y_pred_optimal, average="binary")

    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"\nDefault Threshold (0.5):")
    print(f"  Precision: {p_def:.4f}  Recall: {r_def:.4f}  F1: {f1_def:.4f}")
    print(f"\nOptimal Threshold ({optimal_threshold:.4f}):")
    print(f"  Precision: {p_opt:.4f}  Recall: {r_opt:.4f}  F1: {f1_opt:.4f}")
    print(f"\n🚀 Precision improvement: {((p_opt/p_def - 1)*100):+.1f}%")

    # Save artifacts
    print(f"\n💾 Saving model artifacts to {outdir}...")
    joblib.dump(model, outdir / "model.pkl")
    joblib.dump(scaler, outdir / "scaler.pkl")
    (outdir / "feature_names.json").write_text(json.dumps(features))
    (outdir / "threshold.txt").write_text(str(optimal_threshold))

    # Save comprehensive metrics
    metrics = {
        "average_precision": float(ap),
        "precision": float(p_opt),
        "recall": float(r_opt), 
        "f1": float(f1_opt),
        "threshold": float(optimal_threshold),
        "precision_default": float(p_def),
        "recall_default": float(r_def),
        "f1_default": float(f1_def),
        "precision_improvement": float((p_opt/p_def - 1)*100)
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced fraud detection training with precision optimization")
    parser.add_argument("--data", type=str, required=True, help="Path to creditcard.csv")
    parser.add_argument("--outdir", type=str, default="backend/artifacts")
    parser.add_argument("--model_type", choices=['xgb', 'xgb_tuned', 'ensemble'], default='ensemble')
    parser.add_argument("--resampling", choices=['smote', 'adasyn', 'smoteenn', 'none'], default='smoteenn')
    parser.add_argument("--sampling_strategy", type=float, default=0.3)
    parser.add_argument("--threshold_metric", choices=['f1', 'precision', 'balanced'], default='balanced')
    args = parser.parse_args()
    main(args)