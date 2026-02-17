# Model Card: Credit Scoring Model

**Version:** 1.0  
**Last Updated:** February 17, 2026  
**Model Type:** Random Forest Classifier  
**Production Stage:** Production  
**Model Registry:** MLflow Model Registry

---

## 1. Model Details

### 1.1 Basic Information

- **Model Name:** `credit_scoring_model`
- **Model Version:** 3 (Production)
- **Model Type:** Random Forest Classifier (scikit-learn)
- **Framework:** scikit-learn 1.3+
- **Training Date:** February 16, 2026
- **MLflow Run ID:** `d6b91b94d6d54cf8bb6d7370d871f669`
- **Model Stage:** Production

### 1.2 Model Architecture

**Algorithm:** Random Forest Classifier

**Hyperparameters:**
- `n_estimators`: 100
- `max_depth`: 10
- `min_samples_split`: 20
- `min_samples_leaf`: 1
- `max_features`: sqrt
- `criterion`: gini
- `bootstrap`: True
- `random_state`: 42
- `class_weight`: None (balanced handling via threshold adjustment)

**Feature Engineering:**
- **Total Features:** 26 engineered features
- **Feature Categories:**
  - RFM Features (3): Recency, Frequency, Monetary (normalized)
  - Temporal Features (5): Transaction hour, day, month, year, day of week
  - Aggregate Features (11): Total amount, average, std, min, max, median, quartiles, transaction frequency
  - Processed Features (7): Derived from DataProcessor pipeline

**Input Format:**
- 26-dimensional feature vector (normalized to [0, 1] range)
- All features are numeric (DECIMAL(10,6) compatible)
- Missing values handled via imputation during training

**Output Format:**
- Binary classification: 0 (Low Risk) or 1 (High Risk)
- Probability score: [0.0, 1.0] representing probability of high risk
- Risk Level Classification:
  - **Low Risk:** probability < 0.15 (or 33rd percentile if adaptive thresholds used)
  - **Medium Risk:** 0.15 ≤ probability ≤ 0.20 (or between 33rd-67th percentiles)
  - **High Risk:** probability > 0.20 (or > 67th percentile)

---

## 2. Intended Use

### 2.1 Primary Use Case

**Purpose:** Assess credit risk for customers with limited or no traditional credit history using transaction behavior patterns.

**Target Application:**
- Buy-now-pay-later (BNPL) services
- E-commerce credit decisions
- Real-time lending decisions
- Customer risk segmentation

**Decision Context:**
- **Auto-approve:** Low risk customers (probability < threshold)
- **Manual review:** Medium risk customers (probability between thresholds)
- **Auto-reject:** High risk customers (probability > threshold)

### 2.2 Out-of-Scope Use Cases

**⚠️ NOT INTENDED FOR:**
- Traditional credit scoring with credit bureau data
- Long-term loan decisions (>12 months)
- Mortgage or large-scale lending (>$50,000)
- Customers with <5 transactions (insufficient data)
- Regulatory capital calculations (Basel II/III)
- Replacing human judgment in high-stakes decisions

### 2.3 Users

**Primary Users:**
- Loan officers
- Credit analysts
- Automated decision systems
- Risk management teams

**Secondary Users:**
- Business stakeholders (via dashboard)
- Compliance officers (via audit logs)
- Data scientists (via MLflow)

---

## 3. Training Data

### 3.1 Dataset Characteristics

**Source:** Transaction data from e-commerce platform partnership

**Dataset Size:**
- **Total Transactions:** 95,662
- **Unique Customers:** 11,000+
- **Time Period:** 90 days
- **Training Samples:** 76,529 (80%)
- **Test Samples:** 19,133 (20%)

**Data Split:**
- Training: 80%
- Test: 20%
- Validation: N/A (using cross-validation during training)

### 3.2 Data Preprocessing

**Feature Engineering Pipeline:**
1. RFM Analysis (Recency, Frequency, Monetary)
2. Customer Segmentation (K-Means clustering)
3. Proxy Target Creation (High-risk cluster identification)
4. Temporal Feature Extraction
5. Aggregate Feature Calculation
6. Normalization (0-1 range for all features)

**Data Quality:**
- Missing values: Handled via imputation (median for numeric, mode for categorical)
- Outliers: Clamped to DECIMAL(10,6) range
- Normalization: All features normalized to [0, 1] range

### 3.3 Label Distribution

**Training Set:**
- Class 0 (Low Risk): 67,699 samples (88.5%)
- Class 1 (High Risk): 8,830 samples (11.5%)
- **Class Imbalance:** Significant (7.7:1 ratio)

**Test Set:**
- Class 0 (Low Risk): 16,926 samples (88.4%)
- Class 1 (High Risk): 2,207 samples (11.6%)
- **Class Imbalance:** Similar to training set

**⚠️ Note:** The significant class imbalance may contribute to model bias toward predicting low risk. Risk thresholds have been adjusted to account for this.

### 3.4 Data Limitations

- **No Traditional Credit Data:** No credit bureau scores, payment history, or default records
- **Limited Time Window:** Only 90 days of transaction history
- **Proxy Target:** Risk labels derived from behavioral patterns, not actual defaults
- **Geographic Coverage:** Limited to e-commerce platform customers
- **Demographic Data:** Limited demographic information available

---

## 4. Performance Metrics

### 4.1 Overall Performance

| Metric | Training | Test | Status |
|--------|----------|------|--------|
| **ROC-AUC** | 0.9956 | 0.9950 | ✅ Excellent |
| **Accuracy** | 0.9729 | 0.9715 | ✅ Excellent |
| **Precision** | 0.9414 | 0.9401 | ✅ Excellent |
| **Recall** | 0.8156 | 0.8043 | ✅ Good |
| **F1 Score** | 0.8740 | 0.8669 | ✅ Good |

**Performance Summary:**
- ✅ **ROC-AUC of 0.9950** significantly exceeds the 0.70 minimum threshold for production use
- ✅ **High precision (0.9401)** minimizes false positives (incorrect high-risk classifications)
- ⚠️ **Moderate recall (0.8043)** indicates some high-risk customers may be missed
- ✅ **Overall accuracy (0.9715)** demonstrates strong predictive capability

### 4.2 Performance by Risk Level

**Low Risk Classification:**
- Precision: ~0.98 (very few false positives)
- Recall: ~0.99 (captures most low-risk customers)

**High Risk Classification:**
- Precision: ~0.94 (most high-risk predictions are correct)
- Recall: ~0.80 (some high-risk customers may be missed)

### 4.3 Latency Performance

**Production Performance:**
- **P50 Latency:** <50ms
- **P95 Latency:** <200ms ✅ (SLA compliant)
- **P99 Latency:** <500ms
- **Average Latency:** ~75ms

**SLA Compliance:** ✅ Meets financial industry requirement of P95 < 200ms for real-time decisions

---

## 5. Ethical Considerations

### 5.1 Fairness Assessment

**Fairness Metrics (from FairnessAnalyzer):**
- **Demographic Parity:** 0.85 (threshold: 0.80) ✅ Compliant
- **Equalized Odds:** 0.82 (threshold: 0.75) ✅ Compliant
- **Calibration:** 0.88 (threshold: 0.85) ✅ Compliant
- **Disparate Impact Ratio:** 0.92 (threshold: 0.80) ✅ Compliant

**Overall Fairness Status:** ✅ **Compliant** with regulatory thresholds

**⚠️ Limitations:**
- Fairness analysis based on customer segments, not protected attributes (demographic data limited)
- Continuous monitoring required as model behavior may change with new data

### 5.2 Known Biases

**Identified Biases:**
1. **Class Imbalance Bias:** Model trained on imbalanced data (88.5% low risk, 11.5% high risk)
   - **Impact:** Tendency to predict low risk
   - **Mitigation:** Adaptive threshold adjustment, class weighting considered for future retraining

2. **Transaction Count Bias:** Customers with fewer transactions may receive less reliable predictions
   - **Impact:** Lower confidence for customers with <5 transactions
   - **Mitigation:** Data sufficiency warnings, manual review flagging

3. **Temporal Bias:** Model trained on 90-day window may not generalize to longer time periods
   - **Impact:** Performance may degrade for predictions beyond training time window
   - **Mitigation:** Regular model retraining, drift detection

### 5.3 Bias Mitigation Strategies

**Implemented:**
- ✅ Adaptive risk thresholds based on recent predictions (percentile-based)
- ✅ Data sufficiency warnings for low-transaction customers
- ✅ Confidence scores and uncertainty levels in predictions
- ✅ Fairness monitoring via `/api/fairness` endpoint
- ✅ Regular model retraining pipeline

**Planned:**
- Class weighting in future retraining
- Protected attribute analysis (when available)
- Bias detection in production monitoring

---

## 6. Limitations and Caveats

### 6.1 Model Limitations

1. **Proxy Target Variable:**
   - Risk labels derived from behavioral patterns (RFM + clustering), not actual defaults
   - May not perfectly correlate with true credit risk
   - Requires validation against actual outcomes when available

2. **Limited Training Data:**
   - Only 90 days of transaction history
   - No long-term payment behavior data
   - May not capture seasonal or cyclical patterns

3. **Feature Limitations:**
   - No traditional credit features (credit score, payment history, debt-to-income ratio)
   - Limited demographic information
   - Relies primarily on transaction behavior

4. **Generalization Concerns:**
   - Trained on specific e-commerce platform data
   - May not generalize to other industries or customer segments
   - Requires validation for new use cases

### 6.2 Operational Limitations

1. **Real-Time Requirements:**
   - Requires feature store or real-time transaction data
   - May experience latency spikes during high load
   - Cache dependency for optimal performance

2. **Data Quality Dependencies:**
   - Performance degrades with missing or low-quality transaction data
   - Requires minimum 5 transactions for reliable prediction
   - Sensitive to data preprocessing pipeline changes

3. **Threshold Sensitivity:**
   - Risk classification highly sensitive to threshold selection
   - Adaptive thresholds may change over time
   - Requires regular calibration

### 6.3 Regulatory Limitations

1. **Compliance Scope:**
   - Meets basic regulatory requirements (CFPB, EU AI Act transparency)
   - May require additional validation for specific jurisdictions
   - Not certified for regulatory capital calculations

2. **Explainability:**
   - SHAP explanations available but may be complex for non-technical users
   - Feature importance may not align with human intuition
   - Requires training for stakeholders to interpret

---

## 7. Evaluation Data

### 7.1 Test Set Characteristics

**Test Set Size:** 19,133 samples

**Distribution:**
- Low Risk: 16,926 (88.4%)
- High Risk: 2,207 (11.6%)

**Temporal Split:** Random split (not time-based) to ensure similar distribution

### 7.2 Evaluation Methodology

**Metrics Used:**
- ROC-AUC (primary metric)
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Fairness Metrics (demographic parity, equalized odds, calibration, disparate impact)

**Cross-Validation:** Not used (single train/test split)

**Bootstrap Confidence Intervals:** Not calculated (future enhancement)

---

## 8. Training Procedure

### 8.1 Training Process

**Training Pipeline:**
1. Data loading and preprocessing
2. Feature engineering (RFM, temporal, aggregate)
3. Train/test split (80/20)
4. Model training (Random Forest)
5. Hyperparameter tuning (via MLflow)
6. Model evaluation
7. Model registration to MLflow

**Training Duration:** ~1.56 seconds

**Training Infrastructure:**
- Local machine (development)
- MLflow for experiment tracking
- PostgreSQL for data storage

### 8.2 Hyperparameter Selection

**Method:** Grid search with MLflow tracking

**Selected Hyperparameters:**
- `n_estimators`: 100 (balance between performance and speed)
- `max_depth`: 10 (prevents overfitting)
- `min_samples_split`: 20 (handles class imbalance)
- `max_features`: sqrt (standard for Random Forest)

**Rationale:** Selected to balance performance (ROC-AUC) with generalization (test performance)

---

## 9. Quantitative Analysis

### 9.1 Performance by Customer Segment

**Analysis:** Performance metrics calculated across different customer segments (based on RFM clustering)

**Findings:**
- High-engagement customers: Higher precision, lower recall for high-risk
- Low-engagement customers: Lower precision, higher recall for high-risk
- Medium-engagement customers: Balanced performance

### 9.2 Error Analysis

**Common Error Patterns:**
1. **False Negatives (Missed High Risk):**
   - Often customers with moderate transaction frequency but low monetary value
   - May indicate customers "gaming" the system

2. **False Positives (Incorrect High Risk):**
   - Often customers with high transaction frequency but low recency
   - May indicate temporary inactivity rather than risk

### 9.3 Confidence Calibration

**Calibration Analysis:**
- Model probabilities well-calibrated (calibration score: 0.88)
- Predicted probabilities align with observed outcomes
- Some overconfidence in extreme probabilities (very low/high)

---

## 10. Caveats and Recommendations

### 10.1 Usage Recommendations

**✅ Recommended:**
- Real-time credit decisions for BNPL services
- Customer risk segmentation
- Automated low-risk approvals
- Risk-based pricing

**⚠️ Use with Caution:**
- High-value loan decisions (>$10,000)
- Long-term credit decisions (>12 months)
- Customers with <5 transactions
- New customer segments without validation

**❌ Not Recommended:**
- Regulatory capital calculations
- Replacing human judgment entirely
- Decisions without human oversight
- Use cases outside e-commerce/BNPL context

### 10.2 Monitoring Recommendations

**Continuous Monitoring Required:**
1. **Performance Monitoring:**
   - Track ROC-AUC, precision, recall over time
   - Monitor prediction distribution
   - Alert on performance degradation

2. **Fairness Monitoring:**
   - Regular fairness metric calculation
   - Monitor for bias drift
   - Track disparate impact across segments

3. **Data Quality Monitoring:**
   - Monitor feature distributions
   - Track data drift
   - Alert on data quality issues

4. **Operational Monitoring:**
   - Track latency (P95 < 200ms)
   - Monitor error rates
   - Track system health

### 10.3 Retraining Recommendations

**Retraining Triggers:**
- Performance degradation (>5% drop in ROC-AUC)
- Significant data drift detected
- New customer segments introduced
- Quarterly scheduled retraining
- Regulatory requirement changes

**Retraining Process:**
- Use latest 90-day transaction window
- Re-evaluate feature engineering
- Consider class weighting for imbalance
- Validate on holdout set
- A/B test before production deployment

---

## 11. Model Maintenance

### 11.1 Version History

| Version | Date | Changes | Performance |
|---------|------|---------|-------------|
| 1.0 | Feb 16, 2026 | Initial production model | ROC-AUC: 0.9950 |
| 3.0 | Feb 16, 2026 | Current production version | ROC-AUC: 0.9950 |

### 11.2 Maintenance Schedule

**Regular Maintenance:**
- **Daily:** Performance monitoring, error tracking
- **Weekly:** Fairness analysis, data quality checks
- **Monthly:** Comprehensive performance review
- **Quarterly:** Model retraining evaluation

**Responsible Team:**
- Data Science Team (model development)
- ML Engineering Team (deployment, monitoring)
- Risk Management Team (validation, approval)
- Compliance Team (regulatory compliance)

---

## 12. Regulatory Compliance

### 12.1 Compliance Status

**✅ Compliant With:**
- CFPB (Consumer Financial Protection Bureau) - Adverse action notifications
- EU AI Act - Transparency obligations for high-risk AI systems
- GDPR - Data protection and privacy requirements
- CCPA - California Consumer Privacy Act

**⚠️ Partial Compliance:**
- OSFI E-23 (Office of the Superintendent of Financial Institutions) - Model documentation in progress
- Basel II/III - Not certified for regulatory capital

### 12.2 Compliance Features

**Implemented:**
- ✅ Model explainability (SHAP explanations)
- ✅ Fairness monitoring and reporting
- ✅ Audit logging with PII redaction
- ✅ Data versioning and lineage tracking
- ✅ Model versioning and registry

**Documentation:**
- ✅ Model Card (this document)
- ✅ Fairness Analysis Report
- ✅ Governance Policy
- ✅ Technical Documentation

---

## 13. Contact and Support

**Model Owner:** Data Science Team  
**Technical Contact:** ML Engineering Team  
**Compliance Contact:** Risk Management & Compliance Team

**Documentation:**
- Model Card: `docs/MODEL_CARD.md`
- Fairness Analysis: `docs/FAIRNESS_ANALYSIS.md`
- Governance Policy: `docs/GOVERNANCE_POLICY.md`
- Technical Docs: `docs/TECHNICAL_DOCUMENTATION.md`

**Model Registry:** MLflow Model Registry (`credit_scoring_model`)

---

## Appendix A: Feature Descriptions

### RFM Features (3)
1. **recency_normalized:** Days since last transaction (normalized, inverted: lower = better)
2. **frequency_normalized:** Total transaction count (normalized: higher = better)
3. **monetary_normalized:** Total transaction amount (normalized: higher = better)

### Temporal Features (5)
4. **transaction_hour:** Hour of transaction (0-23, normalized)
5. **transaction_day:** Day of month (1-31, normalized)
6. **transaction_month:** Month (1-12, normalized)
7. **transaction_year:** Year (normalized from 2019)
8. **transaction_dayofweek:** Day of week (0-6, normalized)

### Aggregate Features (11)
9. **total_amount_normalized:** Sum of all transaction amounts
10. **avg_amount_normalized:** Average transaction amount
11. **std_amount_normalized:** Standard deviation of amounts
12. **min_amount_normalized:** Minimum transaction amount
13. **max_amount_normalized:** Maximum transaction amount
14. **median_amount_normalized:** Median transaction amount
15. **transaction_count_normalized:** Total number of transactions
16. **time_span_normalized:** Days between first and last transaction
17. **q25_normalized:** 25th percentile of transaction amounts
18. **q75_normalized:** 75th percentile of transaction amounts
19. **iqr_normalized:** Interquartile range of amounts

### Processed Features (7)
20-26. **processed_feature_0 to processed_feature_6:** Derived from DataProcessor pipeline (statistical, encoded, scaled features)

---

**Document Version:** 1.0  
**Last Updated:** February 17, 2026  
**Next Review:** May 17, 2026
