"""
train.py
Train/compare models, calibrate probabilities, and compute threshold vs net-savings table.
Usage (example):
    python src/train.py --data_path data/Telco-Customer-Churn.csv --out_dir models --test_size 0.25 --n_iter 8
"""
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import joblib
from scipy.stats import randint, uniform

def build_preprocessor(X):
    """Return a ColumnTransformer to preprocess numeric and categorical features."""
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    return pre

def try_import_lightgbm_xgb():
    """Try to import xgboost and lightgbm; return dict telling availability."""
    avail = {'xgb': False, 'lgb': False}
    try:
        import xgboost as xgb
        avail['xgb'] = True
    except Exception:
        pass
    try:
        import lightgbm as lgb
        avail['lgb'] = True
    except Exception:
        pass
    return avail

def main(args):
    df = pd.read_csv(args.data_path)
    # Basic cleaning used in earlier pass
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

    target = 'Churn'
    X = df.drop(columns=[target, 'customerID'], errors='ignore')
    y = df[target].map({'Yes':1,'No':0})

    pre = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=args.test_size, random_state=42)

    # Fit preprocessor once to speed repeated experiments if you want to train classifiers separately
    X_train_trans = pre.fit_transform(X_train)
    X_test_trans = pre.transform(X_test)

    # Define classifiers (simple defaults)
    clfs = {
        'logreg': LogisticRegression(max_iter=2000, solver='liblinear'),
        'rf': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'hgb': HistGradientBoostingClassifier(random_state=42)
    }

    # If available, we will include xgboost/lightgbm (optional)
    avail = try_import_lightgbm_xgb()
    if avail['xgb']:
        import xgboost as xgb
        clfs['xgb'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    if avail['lgb']:
        import lightgbm as lgb
        clfs['lgb'] = lgb.LGBMClassifier(random_state=42)

    # Quick hyperparam spaces for RandomizedSearchCV (user can increase n_iter for deeper search)
    param_spaces = {
        'logreg': {'C': uniform(0.01, 10)},
        'rf': {'n_estimators': randint(100, 400), 'max_depth': randint(4, 40)},
        'hgb': {'max_iter': randint(50, 300)}
    }
    if 'xgb' in clfs:
        param_spaces['xgb'] = {'n_estimators': randint(50, 300), 'max_depth': randint(3,12), 'learning_rate': uniform(0.01,0.4)}
    if 'lgb' in clfs:
        param_spaces['lgb'] = {'n_estimators': randint(50,300), 'num_leaves': randint(15,256), 'learning_rate': uniform(0.01,0.3)}

    results = {}
    best_pipelines = {}

    for name, clf in clfs.items():
        print(f"[train] {name} - starting")
        # Option: run a small randomized search (fast). If args.n_iter==0 we skip tuning.
        if args.n_iter > 0 and name in param_spaces:
            rs = RandomizedSearchCV(clf, param_spaces[name], n_iter=args.n_iter, scoring='roc_auc', cv=2, random_state=42, n_jobs=-1)
            rs.fit(X_train_trans, y_train)
            best_clf = rs.best_estimator_
            print(f"  best params (quick search): {rs.best_params_}, CV AUC: {rs.best_score_:.4f}")
            results[name] = {'cv_best_score': float(rs.best_score_), 'best_params': rs.best_params_}
        else:
            clf.fit(X_train_trans, y_train)
            best_clf = clf
            results[name] = {'cv_best_score': None, 'best_params': None}

        # Build full pipeline (preprocessor + best clf) for saving & serving
        pipeline = Pipeline([('pre', pre), ('clf', best_clf)])
        best_pipelines[name] = pipeline

        # Evaluate on test
        if hasattr(best_clf, "predict_proba"):
            y_proba = best_clf.predict_proba(X_test_trans)[:,1]
        else:
            # fallback: decision function or predict
            try:
                y_proba = best_clf.decision_function(X_test_trans)
            except Exception:
                y_proba = best_clf.predict(X_test_trans)

        y_pred = best_clf.predict(X_test_trans)
        results[name].update({
            'test_accuracy': float(accuracy_score(y_test, y_pred)),
            'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'test_roc_auc': float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
        })

        # Save pipeline
        os.makedirs(args.out_dir, exist_ok=True)
        joblib.dump(pipeline, os.path.join(args.out_dir, f"{name}_pipeline.joblib"))

    # Pick best by test ROC AUC
    best_name = max(results.keys(), key=lambda k: results[k].get('test_roc_auc') or -1)
    print("[train] best model by test ROC AUC:", best_name, results[best_name].get('test_roc_auc'))

    # Calibrate probabilities for the best estimator (conservative approach)
    best_pipe = best_pipelines[best_name]
    best_clf = best_pipe.named_steps['clf']
    calibrated_pipeline = best_pipe
    if hasattr(best_clf, "predict_proba"):
        calibrator = CalibratedClassifierCV(best_clf, cv=3, method='isotonic')
        # NOTE: we must pass the transformed X_train arrays when fitting the calibrator
        X_train_trans = best_pipe.named_steps['pre'].transform(X_train)
        calibrator.fit(X_train_trans, y_train)
        calibrated_pipeline = Pipeline([('pre', best_pipe.named_steps['pre']), ('clf', calibrator)])
        joblib.dump(calibrated_pipeline, os.path.join(args.out_dir, f"{best_name}_calibrated_pipeline.joblib"))
        print("[train] calibrated pipeline saved:", f"{best_name}_calibrated_pipeline.joblib")
    else:
        print("[train] best model does not support predict_proba; skipping calibration")

    # ------------------ ROI: threshold vs net-savings ------------------
    # Use baseline LTV estimate (you can replace with survival analysis later)
    avg_monthly_churners = df.loc[df['Churn']=='Yes','MonthlyCharges'].mean()
    retained_avg_tenure = df.loc[df['Churn']=='No','tenure'].mean()
    baseline_ltv = avg_monthly_churners * retained_avg_tenure

    # get probabilities on test set using calibrated pipeline if available
    pipe_for_probs = calibrated_pipeline
    X_test_full = X_test  # still in original form; pipeline transforms internally
    y_proba_test = pipe_for_probs.predict_proba(X_test_full)[:,1]

    thresholds = np.linspace(0.01, 0.99, 99)
    net_rows = []
    for thr in thresholds:
        preds = (y_proba_test >= thr).astype(int)
        tp = int(((preds==1) & (y_test==1)).sum())
        fp = int(((preds==1) & (y_test==0)).sum())
        total_targeted = tp + fp
        expected_retained = args.conversion * tp  # only prevents churn among true churners
        gross = expected_retained * baseline_ltv
        cost = total_targeted * args.cost_per_target
        net = gross - cost
        net_rows.append({'threshold': float(thr), 'tp': tp, 'fp': fp, 'total_targeted': total_targeted,
                         'gross_savings': gross, 'campaign_cost': cost, 'net_savings': net})

    net_df = pd.DataFrame(net_rows)
    os.makedirs(args.report_dir, exist_ok=True)
    net_df.to_csv(os.path.join(args.report_dir, 'threshold_net_savings.csv'), index=False)

    # Save results meta
    with open(os.path.join(args.report_dir, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("[train] Done. Artifacts saved to:", args.out_dir, "and", args.report_dir)
    print("[train] Best threshold (by net) example:",
          net_df.loc[net_df['net_savings'].idxmax()].to_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--out_dir', default='models')
    parser.add_argument('--report_dir', default='reports')
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--n_iter', type=int, default=0, help='RandomizedSearchCV n_iter; set >0 to tune (costly)')
    parser.add_argument('--cost_per_target', type=float, default=50.0)
    parser.add_argument('--conversion', type=float, default=0.20)
    args = parser.parse_args()
    main(args)
