# Bati Bank Credit Scoring MLOps

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

---

## Task 1: Credit Scoring Business Understanding

This section addresses three critical questions about credit risk modeling in a regulated financial environment.

### 1. Basel II Accord's Emphasis on Risk Measurement and Model Interpretability

The Basel II Capital Accord fundamentally transformed banking regulation by introducing a risk-based approach to capital adequacy. Under Basel II, banks are required to:

- **Quantify credit risk** with sufficient precision to determine appropriate capital reserves
- **Document model methodologies** comprehensively for regulatory review and audit
- **Demonstrate model validation** through backtesting, stress testing, and ongoing monitoring
- **Maintain transparency** in risk assessment processes to satisfy both regulators and internal risk management

This regulatory framework directly influences our model development approach:

**Interpretability Requirement**: Basel II mandates that banks understand and explain their risk models. An interpretable model (e.g., Logistic Regression with Weight of Evidence transformations) allows risk officers, auditors, and regulators to:
- Trace how individual features contribute to risk predictions
- Validate that model behavior aligns with business logic and domain expertise
- Identify potential biases or discriminatory patterns
- Explain credit decisions to customers and stakeholders

**Documentation Necessity**: Comprehensive documentation is not optional—it's a regulatory requirement. Our model must include:
- Clear feature definitions and transformations
- Model performance metrics and validation results
- Documentation of assumptions and limitations
- Procedures for model monitoring and recalibration

**Capital Adequacy Implications**: The accuracy and reliability of our risk probability estimates directly impact Bati Bank's capital allocation. Overestimating risk leads to excessive capital reserves (reducing profitability), while underestimating risk exposes the bank to regulatory penalties and financial losses.

### 2. Proxy Variable Necessity and Associated Business Risks

**Why Proxy Variables Are Necessary**:

In this project, we lack direct historical default data because:
- The eCommerce partner is a new entity without established credit history
- Traditional credit bureau data may not be available for all customers
- The buy-now-pay-later service is novel, so historical performance data doesn't exist

We must create a **proxy variable** using available behavioral data (RFM patterns—Recency, Frequency, Monetary) to approximate credit risk. This proxy might be defined as:
- **High Risk (Bad)**: Customers with patterns indicating potential default (e.g., irregular payment behavior, declining transaction frequency, increasing transaction amounts relative to historical patterns)
- **Low Risk (Good)**: Customers with stable, positive behavioral patterns

**Potential Business Risks of Proxy-Based Predictions**:

1. **Model Risk**: The proxy may not accurately reflect true credit risk. RFM patterns from eCommerce transactions may correlate with creditworthiness, but the relationship is indirect and may break down under different economic conditions or customer lifecycle stages.

2. **Concept Drift**: Customer behavior in eCommerce may change over time, causing the proxy relationship to degrade. A model trained on pre-pandemic behavior may not generalize to post-pandemic patterns.

3. **Selection Bias**: The proxy definition itself introduces bias. If we define "bad" customers too narrowly, we may miss emerging risk patterns. If defined too broadly, we may reject creditworthy customers.

4. **Regulatory Scrutiny**: Regulators may question the validity of proxy-based models, especially if they cannot be validated against actual default outcomes. This could lead to:
   - Rejection of the model for regulatory capital calculations
   - Requirement for additional validation studies
   - Mandatory conservative adjustments to risk estimates

5. **Business Impact**: Incorrect proxy definitions can lead to:
   - **Type I Errors**: Rejecting creditworthy customers (lost revenue, customer dissatisfaction)
   - **Type II Errors**: Approving high-risk customers (increased defaults, financial losses)

6. **Limited Validation**: Without true default labels, we cannot directly validate model performance on the actual outcome of interest. We must rely on indirect validation methods, which are inherently less reliable.

**Mitigation Strategies**:
- Use multiple validation techniques (cross-validation, temporal validation, business logic checks)
- Implement conservative risk estimates initially, then refine based on observed outcomes
- Establish continuous monitoring to detect when proxy relationships degrade
- Document all assumptions and limitations transparently

### 3. Trade-offs: Simple Interpretable Models vs. Complex High-Performance Models

In a regulated financial context, the choice between Logistic Regression with WoE (Weight of Evidence) and Gradient Boosting involves critical trade-offs:

#### Logistic Regression with WoE - Advantages:
- **Regulatory Compliance**: Highly interpretable—each feature's contribution is transparent and explainable
- **Audit Trail**: Clear documentation of how each variable affects the risk score
- **Business Alignment**: WoE transformations align with credit scoring industry standards
- **Stability**: Less prone to overfitting, more stable predictions across different data distributions
- **Regulatory Acceptance**: Well-understood by regulators and risk officers, easier to gain approval
- **Feature Engineering Discipline**: WoE requires careful binning and transformation, enforcing domain expertise

#### Logistic Regression with WoE - Disadvantages:
- **Limited Complexity**: Cannot capture complex non-linear interactions automatically
- **Lower Performance**: May achieve lower AUC/accuracy compared to ensemble methods
- **Manual Feature Engineering**: Requires significant domain expertise and time investment
- **Limited Scalability**: Adding many features can become unwieldy

#### Gradient Boosting (XGBoost/LightGBM) - Advantages:
- **Higher Performance**: Typically achieves superior predictive accuracy and AUC scores
- **Automatic Feature Interactions**: Captures complex non-linear relationships without explicit engineering
- **Handles Non-linearity**: Better at modeling complex patterns in data
- **Feature Importance**: Provides feature importance metrics (though less interpretable than coefficients)

#### Gradient Boosting - Disadvantages:
- **Black Box Nature**: Difficult to explain individual predictions or feature contributions
- **Regulatory Challenges**: May face resistance from regulators who require model interpretability
- **Overfitting Risk**: More prone to overfitting, especially with limited data
- **Validation Complexity**: Harder to validate that model behavior aligns with business logic
- **Bias Detection**: More difficult to identify and mitigate potential discriminatory patterns

#### Key Trade-offs in Regulated Context:

1. **Regulatory Approval**: 
   - Simple models: Easier to gain regulatory approval, faster time-to-market
   - Complex models: May require extensive documentation, additional validation, potential rejection

2. **Business Trust**:
   - Simple models: Risk officers and business stakeholders can understand and trust the model
   - Complex models: May face skepticism and require extensive education

3. **Performance vs. Interpretability**:
   - Simple models: Lower performance but higher trust and compliance
   - Complex models: Higher performance but potential regulatory and trust issues

4. **Operational Risk**:
   - Simple models: Easier to monitor, debug, and maintain
   - Complex models: Harder to diagnose issues, more complex monitoring requirements

5. **Customer Communication**:
   - Simple models: Can explain credit decisions to customers ("Your score was low due to X, Y, Z")
   - Complex models: Difficult to provide meaningful explanations

#### Recommended Approach for This Project:

Given the regulatory context and business requirements, a **hybrid approach** may be optimal:

1. **Primary Model**: Start with Logistic Regression + WoE for regulatory compliance and interpretability
2. **Validation Model**: Use Gradient Boosting as a benchmark to validate that the simpler model isn't missing critical patterns
3. **Ensemble Consideration**: If performance gap is significant, consider a carefully documented ensemble with clear interpretability components
4. **Documentation**: Regardless of model choice, maintain comprehensive documentation of:
   - Model selection rationale
   - Performance metrics
   - Interpretability analysis
   - Validation procedures
   - Limitations and assumptions

**Conclusion**: In regulated financial contexts, interpretability and regulatory compliance often outweigh marginal performance gains. However, the optimal choice depends on the specific regulatory environment, business requirements, and the magnitude of the performance difference between models.
