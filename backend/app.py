import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import logging

from utils import load_artifacts, prepare_vector, get_threshold, top_shap_reasons

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aegis AI – Fraud Shield", 
    version="1.0.0",
    description="Enterprise fraud detection with 80.2% precision and explainable AI",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve static files in production
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

_model, _scaler, _feature_names, _explainer = load_artifacts()

class Transaction(BaseModel):
    # Accept flexible payload – any numeric fields matching feature_names
    payload: Dict[str, float]

@app.get("/")
def root():
    return {
        "message": "Aegis AI Fraud Detection API",
        "version": "1.0.0",
        "status": "operational",
        "precision": "80.2%",
        "docs": "/docs" if os.getenv("ENVIRONMENT") != "production" else "disabled"
    }

@app.get("/health")
def health():
    try:
        # Verify model is loaded
        if _model is None or _feature_names is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "features_count": len(_feature_names),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics")
def get_metrics():
    """Get model performance metrics for business impact display"""
    import json
    from pathlib import Path
    
    metrics_path = Path(__file__).parent / "artifacts" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        
        # Industry-standard assumptions for mid-size bank
        daily_transactions = 50000          # Mid-size bank volume
        fraud_rate = 0.001                  # 0.1% industry average fraud rate
        avg_fraud_amount = 1200             # Industry average fraud amount
        manual_review_cost = 25             # Cost per manual review
        
        # Calculate fraud detection impact
        daily_fraud_attempts = daily_transactions * fraud_rate
        prevented_fraud = daily_fraud_attempts * metrics.get("recall", 0.827)
        fraud_savings_per_day = prevented_fraud * avg_fraud_amount
        
        # Calculate false positive reduction impact
        current_precision = metrics.get("precision", 0.802)
        baseline_precision = metrics.get("precision_default", 0.242)
        
        # False positive rates
        current_fp_rate = 1 - current_precision
        baseline_fp_rate = 1 - baseline_precision
        fp_reduction = (baseline_fp_rate - current_fp_rate) / baseline_fp_rate
        
        # Daily false positives saved
        daily_fp_saved = daily_transactions * baseline_fp_rate * fp_reduction
        review_savings_per_day = daily_fp_saved * manual_review_cost
        
        # Total daily and annual savings
        total_daily_savings = fraud_savings_per_day + review_savings_per_day
        annual_savings = total_daily_savings * 365
        
        return {
            **metrics,
            "business_impact": {
                "daily_transactions_processed": daily_transactions,
                "fraud_attempts_detected": round(prevented_fraud, 1),
                "fraud_prevented_per_day": f"${round(fraud_savings_per_day):,}",
                "false_positives_reduced": f"{fp_reduction*100:.1f}%",
                "manual_reviews_saved": round(daily_fp_saved),
                "review_cost_savings": f"${round(review_savings_per_day):,}",
                "total_daily_savings": f"${round(total_daily_savings):,}",
                "annual_savings": f"${round(annual_savings/1000000, 1)}M",
                "detection_accuracy": f"{current_precision * 100:.1f}%"
            }
        }
    return {"error": "Metrics not available"}

def get_risk_category(fraud_probability: float):
    """Categorize risk based on fraud probability"""
    prob_percent = fraud_probability * 100
    
    if prob_percent < 30:
        return {
            "category": "LOW",
            "label": "APPROVED",
            "color": "green",
            "icon": "✅",
            "action": "Process automatically",
            "message": "Low risk. Safe to process.",
            "priority": 1
        }
    elif prob_percent < 70:
        return {
            "category": "MEDIUM", 
            "label": "REVIEW",
            "color": "orange",
            "icon": "⚠️",
            "action": "Manual review recommended",
            "message": "Medium risk. Consider additional verification.",
            "priority": 2
        }
    elif prob_percent < 90:
        return {
            "category": "HIGH",
            "label": "BLOCK",
            "color": "red", 
            "icon": "🚨",
            "action": "Block and investigate",
            "message": "High fraud risk. Transaction blocked for review.",
            "priority": 3
        }
    else:
        return {
            "category": "CRITICAL",
            "label": "FRAUD",
            "color": "darkred",
            "icon": "🔴",
            "action": "Immediate block and alert",
            "message": "Critical fraud alert. Immediate investigation required.",
            "priority": 4
        }

@app.post("/predict")
def predict(tx: Transaction):
    try:
        logger.info(f"Processing transaction with {len(tx.payload)} features")
        
        x = prepare_vector(tx.payload, _feature_names, _scaler)
        proba = float(_model.predict_proba(x)[0, 1])
        risk_info = get_risk_category(proba)
        reasons = top_shap_reasons(x, _feature_names, _explainer, top_k=3)
        
        # Get transaction amount for business impact
        transaction_amount = tx.payload.get('Amount', 0)
        
        result = {
            "label": risk_info["label"],
            "risk_category": risk_info["category"],
            "probability_fraud": proba,
            "risk_info": risk_info,
            "reasons": reasons,
            "transaction_amount": transaction_amount,
            "business_impact": {
                "potential_loss": transaction_amount if risk_info["priority"] >= 3 else 0,
                "action_required": risk_info["action"],
                "review_priority": risk_info["priority"]
            }
        }
        
        logger.info(f"Prediction completed: {risk_info['category']} risk ({proba:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction service error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting Aegis AI Fraud Detection API on {host}:{port}")
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
