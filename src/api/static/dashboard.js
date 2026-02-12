/**
 * Credit Scoring Dashboard - Interactive Interface
 * 
 * Provides a user-friendly interface for loan officers and credit analysts
 * to explore risk profiles, test scenarios, and understand predictions.
 */

class CreditScoringDashboard {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.featureNames = [];
        this.currentPrediction = null;
        this.currentExplanation = null;
        this.init();
    }

    async init() {
        await this.loadFeatureNames();
        this.setupEventListeners();
        this.loadSampleData();
    }

    async loadFeatureNames() {
        try {
            // Try to get feature names from a dedicated endpoint or use defaults
            const response = await fetch(`${this.apiBaseUrl}/api/feature-names`);
            if (response.ok) {
                const data = await response.json();
                this.featureNames = data.feature_names || this.getDefaultFeatureNames();
            } else {
                this.featureNames = this.getDefaultFeatureNames();
            }
        } catch (error) {
            console.warn('Could not load feature names, using defaults:', error);
            this.featureNames = this.getDefaultFeatureNames();
        }
        this.renderFeatureInputs();
    }

    getDefaultFeatureNames() {
        // Default feature names based on the model
        return [
            'RFM_Recency', 'RFM_Frequency', 'RFM_Monetary',
            'Transaction_Count', 'Avg_Transaction_Amount', 'Total_Spend',
            'Days_Since_First_Transaction', 'Days_Since_Last_Transaction',
            'Transaction_Hour_Mean', 'Transaction_Day_Mean',
            'Category_Diversity', 'Channel_Diversity',
            'Weekend_Transaction_Ratio', 'Evening_Transaction_Ratio',
            'High_Value_Transaction_Ratio', 'Refund_Ratio',
            'Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3',
            'WoE_Feature_1', 'WoE_Feature_2', 'WoE_Feature_3',
            'WoE_Feature_4', 'WoE_Feature_5', 'WoE_Feature_6'
        ];
    }

    renderFeatureInputs() {
        const container = document.getElementById('feature-inputs');
        container.innerHTML = '';
        
        // Group features into sections
        const sections = this.groupFeatures();
        
        Object.entries(sections).forEach(([sectionName, features]) => {
            const section = document.createElement('div');
            section.className = 'feature-section';
            section.innerHTML = `
                <h3>${sectionName}</h3>
                <div class="feature-grid">
                    ${features.map((name, idx) => {
                        const globalIdx = this.featureNames.indexOf(name);
                        return `
                            <div class="feature-input-group">
                                <label for="feature-${globalIdx}">${name}</label>
                                <input 
                                    type="number" 
                                    id="feature-${globalIdx}" 
                                    data-feature-index="${globalIdx}"
                                    step="0.0001"
                                    placeholder="0.0"
                                    class="feature-input"
                                />
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
            container.appendChild(section);
        });
    }

    groupFeatures() {
        const groups = {
            'RFM Metrics': [],
            'Transaction Patterns': [],
            'Temporal Features': [],
            'Customer Segments': [],
            'WoE Features': [],
            'Other Features': []
        };

        this.featureNames.forEach((name, idx) => {
            const lower = name.toLowerCase();
            if (lower.includes('rfm')) {
                groups['RFM Metrics'].push(name);
            } else if (lower.includes('transaction') || lower.includes('spend') || lower.includes('amount')) {
                groups['Transaction Patterns'].push(name);
            } else if (lower.includes('hour') || lower.includes('day') || lower.includes('weekend') || lower.includes('evening')) {
                groups['Temporal Features'].push(name);
            } else if (lower.includes('cluster')) {
                groups['Customer Segments'].push(name);
            } else if (lower.includes('woe')) {
                groups['WoE Features'].push(name);
            } else {
                groups['Other Features'].push(name);
            }
        });

        // Remove empty groups
        Object.keys(groups).forEach(key => {
            if (groups[key].length === 0) delete groups[key];
        });

        return groups;
    }

    setupEventListeners() {
        document.getElementById('predict-btn').addEventListener('click', () => this.predict());
        document.getElementById('explain-btn').addEventListener('click', () => this.explain());
        document.getElementById('load-sample-btn').addEventListener('click', () => this.loadSampleData());
        document.getElementById('clear-btn').addEventListener('click', () => this.clearForm());
        document.getElementById('scenario-test-btn').addEventListener('click', () => this.showScenarioTester());
    }

    loadSampleData() {
        // Load sample feature values
        const sampleFeatures = [
            0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994,
            -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ];

        sampleFeatures.forEach((value, idx) => {
            const input = document.getElementById(`feature-${idx}`);
            if (input) {
                input.value = value;
            }
        });
    }

    clearForm() {
        document.querySelectorAll('.feature-input').forEach(input => {
            input.value = '';
        });
        this.clearResults();
    }

    clearResults() {
        document.getElementById('prediction-result').style.display = 'none';
        document.getElementById('explanation-result').style.display = 'none';
        this.currentPrediction = null;
        this.currentExplanation = null;
    }

    getFeatureValues() {
        const values = [];
        for (let i = 0; i < this.featureNames.length; i++) {
            const input = document.getElementById(`feature-${i}`);
            const value = input ? parseFloat(input.value) || 0 : 0;
            values.push(value);
        }
        return values;
    }

    async predict() {
        const features = this.getFeatureValues();
        const loadingEl = document.getElementById('loading');
        const resultEl = document.getElementById('prediction-result');

        try {
            loadingEl.style.display = 'block';
            resultEl.style.display = 'none';

            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    features: features,
                    include_explanation: true 
                })
            });

            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.statusText}`);
            }

            const data = await response.json();
            this.currentPrediction = data;
            this.displayPrediction(data);
            
            // If explanation is included, display it
            if (data.explanation) {
                this.currentExplanation = data.explanation;
                this.displayExplanation(data.explanation);
            }

        } catch (error) {
            this.showError('Prediction failed: ' + error.message);
        } finally {
            loadingEl.style.display = 'none';
        }
    }

    async explain() {
        const features = this.getFeatureValues();
        const loadingEl = document.getElementById('loading');
        const resultEl = document.getElementById('explanation-result');

        try {
            loadingEl.style.display = 'block';
            resultEl.style.display = 'none';

            const response = await fetch(`${this.apiBaseUrl}/explain?include_plot=false`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: features })
            });

            if (!response.ok) {
                throw new Error(`Explanation failed: ${response.statusText}`);
            }

            const data = await response.json();
            this.currentExplanation = data;
            this.displayExplanation(data);

        } catch (error) {
            this.showError('Explanation failed: ' + error.message);
        } finally {
            loadingEl.style.display = 'none';
        }
    }

    displayPrediction(data) {
        const resultEl = document.getElementById('prediction-result');
        const riskLevel = data.risk_level;
        const riskClass = riskLevel === 'low' ? 'low-risk' : riskLevel === 'high' ? 'high-risk' : 'medium-risk';

        resultEl.innerHTML = `
            <div class="prediction-card ${riskClass}">
                <h3>Prediction Result</h3>
                <div class="prediction-main">
                    <div class="risk-level ${riskClass}">
                        <span class="risk-label">${riskLevel.toUpperCase()} RISK</span>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${data.probability * 100}%"></div>
                        </div>
                        <div class="probability-text">${(data.probability * 100).toFixed(2)}%</div>
                    </div>
                    <div class="prediction-details">
                        <div class="detail-item">
                            <span class="label">Prediction:</span>
                            <span class="value">${data.prediction === 1 ? 'High Risk' : 'Low Risk'}</span>
                        </div>
                        <div class="detail-item">
                            <span class="label">Risk Level:</span>
                            <span class="value ${riskClass}">${riskLevel}</span>
                        </div>
                        <div class="detail-item">
                            <span class="label">Probability:</span>
                            <span class="value">${(data.probability * 100).toFixed(2)}%</span>
                        </div>
                        <div class="recommendation">
                            ${this.getRecommendation(riskLevel)}
                        </div>
                    </div>
                </div>
            </div>
        `;
        resultEl.style.display = 'block';
    }

    getRecommendation(riskLevel) {
        const recommendations = {
            'low': '✅ <strong>Recommendation:</strong> Auto-approve. Customer shows low risk indicators.',
            'medium': '⚠️ <strong>Recommendation:</strong> Manual review required. Additional verification may be needed.',
            'high': '❌ <strong>Recommendation:</strong> Auto-reject. Customer shows high risk indicators.'
        };
        return `<div class="recommendation-text">${recommendations[riskLevel] || ''}</div>`;
    }

    displayExplanation(explanation) {
        const resultEl = document.getElementById('explanation-result');
        
        // Sort features by absolute SHAP value
        const topFeatures = explanation.feature_importance.slice(0, 10);
        
        resultEl.innerHTML = `
            <div class="explanation-card">
                <h3>Model Explanation</h3>
                <div class="explanation-summary">
                    <p>${explanation.explanation_summary || 'Explanation generated successfully.'}</p>
                </div>
                <div class="shap-visualization">
                    <h4>Top Contributing Features</h4>
                    <div class="shap-bars">
                        ${topFeatures.map(feat => {
                            const isPositive = feat.shap_value > 0;
                            const absValue = Math.abs(feat.shap_value);
                            const percentage = (absValue / Math.max(...topFeatures.map(f => Math.abs(f.shap_value)))) * 100;
                            return `
                                <div class="shap-bar-item">
                                    <div class="shap-bar-label">
                                        <span>${feat.feature}</span>
                                        <span class="shap-value ${isPositive ? 'positive' : 'negative'}">
                                            ${isPositive ? '+' : ''}${feat.shap_value.toFixed(4)}
                                        </span>
                                    </div>
                                    <div class="shap-bar-container">
                                        <div 
                                            class="shap-bar ${isPositive ? 'positive' : 'negative'}" 
                                            style="width: ${percentage}%"
                                        ></div>
                                    </div>
                                    <div class="feature-value">Value: ${feat.feature_value.toFixed(4)}</div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
                <div class="explanation-details">
                    <div class="detail-row">
                        <span>Base Value:</span>
                        <span>${explanation.base_value?.toFixed(4) || 'N/A'}</span>
                    </div>
                </div>
            </div>
        `;
        resultEl.style.display = 'block';
    }

    showScenarioTester() {
        // Create modal for scenario testing
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <h2>Scenario Testing</h2>
                <p>Adjust feature values to see how they affect the risk prediction.</p>
                <div id="scenario-controls"></div>
                <button id="test-scenario-btn" class="btn btn-primary">Test Scenario</button>
                <div id="scenario-result"></div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Setup scenario controls
        this.setupScenarioControls(modal);
        
        // Close modal
        modal.querySelector('.close-modal').addEventListener('click', () => {
            modal.remove();
        });
    }

    setupScenarioControls(modal) {
        const controls = modal.querySelector('#scenario-controls');
        const topFeatures = this.currentExplanation?.feature_importance?.slice(0, 5) || [];
        
        if (topFeatures.length === 0) {
            controls.innerHTML = '<p>Please run an explanation first to see which features to test.</p>';
            return;
        }

        controls.innerHTML = topFeatures.map(feat => {
            const currentValue = feat.feature_value;
            return `
                <div class="scenario-control">
                    <label>${feat.feature}</label>
                    <input 
                        type="range" 
                        min="${currentValue - 2}" 
                        max="${currentValue + 2}" 
                        step="0.1"
                        value="${currentValue}"
                        data-feature-name="${feat.feature}"
                        class="scenario-slider"
                    />
                    <span class="scenario-value">${currentValue.toFixed(4)}</span>
                </div>
            `;
        }).join('');

        // Update value display on slider change
        controls.querySelectorAll('.scenario-slider').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const valueSpan = e.target.parentElement.querySelector('.scenario-value');
                valueSpan.textContent = parseFloat(e.target.value).toFixed(4);
            });
        });

        // Test scenario
        modal.querySelector('#test-scenario-btn').addEventListener('click', async () => {
            const features = this.getFeatureValues();
            
            // Update features based on sliders
            controls.querySelectorAll('.scenario-slider').forEach(slider => {
                const featureName = slider.dataset.featureName;
                const idx = this.featureNames.indexOf(featureName);
                if (idx >= 0) {
                    features[idx] = parseFloat(slider.value);
                }
            });

            // Run prediction
            try {
                const response = await fetch(`${this.apiBaseUrl}/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: features, include_explanation: true })
                });
                const data = await response.json();
                
                const resultDiv = modal.querySelector('#scenario-result');
                resultDiv.innerHTML = `
                    <div class="scenario-result-card">
                        <h4>Scenario Result</h4>
                        <div class="risk-comparison">
                            <div>Previous: ${this.currentPrediction?.risk_level || 'N/A'}</div>
                            <div>New: <strong>${data.risk_level}</strong></div>
                            <div>Probability Change: ${((data.probability - (this.currentPrediction?.probability || 0)) * 100).toFixed(2)}%</div>
                        </div>
                    </div>
                `;
            } catch (error) {
                alert('Scenario test failed: ' + error.message);
            }
        });
    }

    showError(message) {
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.textContent = message;
        document.body.appendChild(errorEl);
        setTimeout(() => errorEl.remove(), 5000);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new CreditScoringDashboard();
});
