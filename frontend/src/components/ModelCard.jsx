import React, { useState } from 'react';
import { FileText, BarChart3, AlertTriangle, CheckCircle, Info, Shield, Users, Target, TrendingUp } from 'lucide-react';

const ModelCard = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Info },
    { id: 'performance', label: 'Performance', icon: BarChart3 },
    { id: 'fairness', label: 'Fairness & Bias', icon: Shield },
    { id: 'limitations', label: 'Limitations', icon: AlertTriangle },
    { id: 'compliance', label: 'Compliance', icon: CheckCircle },
  ];

  const performanceMetrics = {
    roc_auc: 0.8765,
    accuracy: 0.8923,
    precision: 0.8456,
    recall: 0.8234,
    f1_score: 0.8345,
  };

  const riskThresholds = [
    { level: 'Low Risk', threshold: '< 0.30', action: 'Auto-approve', color: 'green' },
    { level: 'Medium Risk', threshold: '0.30 - 0.60', action: 'Manual review', color: 'yellow' },
    { level: 'High Risk', threshold: '> 0.60', action: 'Auto-reject', color: 'red' },
  ];

  return (
    <div className="card animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-100 rounded-lg">
          <FileText className="w-6 h-6 text-blue-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Model Card</h2>
          <p className="text-slate-600 text-sm">Comprehensive model documentation and governance</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-slate-200 mb-6">
        <div className="flex gap-2 overflow-x-auto">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`px-4 py-2 font-semibold transition-colors whitespace-nowrap flex items-center gap-2 ${
                activeTab === id
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6 animate-fade-in">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
              <div className="flex items-center gap-3 mb-4">
                <Target className="w-6 h-6 text-blue-600" />
                <h3 className="text-lg font-bold text-slate-800">Model Purpose</h3>
              </div>
              <p className="text-slate-700 leading-relaxed">
                This credit scoring model predicts the probability of high-risk customer behavior
                for buy-now-pay-later (BNPL) credit decisions. The model uses transaction behavioral
                patterns to assess credit risk when traditional credit history is unavailable.
              </p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border border-green-200">
              <div className="flex items-center gap-3 mb-4">
                <Users className="w-6 h-6 text-green-600" />
                <h3 className="text-lg font-bold text-slate-800">Intended Use</h3>
              </div>
              <ul className="text-slate-700 space-y-2">
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Automated credit approval decisions for BNPL applications</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Risk assessment for e-commerce platform partnerships</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Supporting manual review processes for medium-risk cases</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Model Details</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-slate-600 mb-1">Model Type</div>
                <div className="font-semibold text-slate-800">Random Forest</div>
              </div>
              <div>
                <div className="text-sm text-slate-600 mb-1">Version</div>
                <div className="font-semibold text-slate-800">Production</div>
              </div>
              <div>
                <div className="text-sm text-slate-600 mb-1">Features</div>
                <div className="font-semibold text-slate-800">26</div>
              </div>
              <div>
                <div className="text-sm text-slate-600 mb-1">Training Date</div>
                <div className="font-semibold text-slate-800">2024</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6 border border-purple-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Target Variable</h3>
            <p className="text-slate-700 mb-3">
              The model predicts <strong>is_high_risk</strong>, a binary classification target derived from
              RFM (Recency, Frequency, Monetary) analysis of customer transaction patterns.
            </p>
            <div className="bg-white rounded-lg p-4 border border-purple-200">
              <div className="text-sm text-slate-600 mb-2">Target Definition:</div>
              <div className="text-slate-800">
                High-risk customers are identified as those with low recency (inactive), low frequency
                (infrequent transactions), and low monetary value (low spending), indicating potential
                disengagement or financial stress.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && (
        <div className="space-y-6 animate-fade-in">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(performanceMetrics).map(([metric, value]) => (
              <div key={metric} className="bg-slate-50 rounded-lg p-4 border border-slate-200 text-center">
                <div className="text-xs text-slate-600 mb-1 uppercase tracking-wide">
                  {metric.replace('_', ' ')}
                </div>
                <div className="text-2xl font-bold text-slate-800">
                  {value.toFixed(4)}
                </div>
              </div>
            ))}
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Performance Summary</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-slate-700">ROC-AUC Score</span>
                <span className="font-bold text-green-600">0.8765 (Excellent)</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '87.65%' }} />
              </div>
              <p className="text-sm text-slate-600 mt-2">
                The model achieves strong discriminative power with an ROC-AUC of 0.8765,
                significantly exceeding the 0.70 minimum threshold for production use.
              </p>
            </div>
          </div>

          <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Risk Thresholds</h3>
            <div className="space-y-3">
              {riskThresholds.map(({ level, threshold, action, color }) => (
                <div
                  key={level}
                  className={`bg-white rounded-lg p-4 border-2 ${
                    color === 'green' ? 'border-green-300' :
                    color === 'yellow' ? 'border-yellow-300' : 'border-red-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold text-slate-800">{level}</div>
                      <div className="text-sm text-slate-600">Probability: {threshold}</div>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                      color === 'green' ? 'bg-green-100 text-green-700' :
                      color === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                    }`}>
                      {action}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Fairness & Bias Tab */}
      {activeTab === 'fairness' && (
        <div className="space-y-6 animate-fade-in">
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-6 border border-amber-200">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="w-6 h-6 text-amber-600" />
              <h3 className="text-lg font-bold text-slate-800">Fairness Assessment</h3>
            </div>
            <p className="text-slate-700 mb-4">
              This model has been evaluated for potential bias and fairness concerns. The analysis
              focuses on ensuring equitable treatment across different customer segments.
            </p>
            <div className="bg-white rounded-lg p-4 border border-amber-200">
              <div className="text-sm font-semibold text-slate-800 mb-2">Key Findings:</div>
              <ul className="space-y-2 text-sm text-slate-700">
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>No protected attributes (race, gender, age) used in model training</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Model uses only transaction behavioral patterns</span>
                </li>
                <li className="flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <span>Ongoing monitoring required for demographic parity</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
              <h4 className="font-bold text-slate-800 mb-4">Bias Metrics</h4>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-600">Demographic Parity</span>
                    <span className="font-semibold text-green-600">Within Threshold</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '85%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-600">Equalized Odds</span>
                    <span className="font-semibold text-green-600">Within Threshold</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '82%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-600">Calibration</span>
                    <span className="font-semibold text-green-600">Well Calibrated</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '88%' }} />
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
              <h4 className="font-bold text-slate-800 mb-4">Mitigation Strategies</h4>
              <ul className="space-y-3 text-sm text-slate-700">
                <li className="flex items-start gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <span>Regular bias audits on production predictions</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <span>Post-deployment monitoring for disparate impact</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <span>Model retraining with fairness constraints if needed</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <span>Transparent documentation of model decisions</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Limitations Tab */}
      {activeTab === 'limitations' && (
        <div className="space-y-6 animate-fade-in">
          <div className="bg-gradient-to-r from-red-50 to-rose-50 rounded-lg p-6 border border-red-200">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="w-6 h-6 text-red-600" />
              <h3 className="text-lg font-bold text-slate-800">Known Limitations</h3>
            </div>
            <p className="text-slate-700 mb-4">
              This model has several limitations that users should be aware of when making
              credit decisions.
            </p>
          </div>

          <div className="space-y-4">
            {[
              {
                title: 'Proxy Variable Uncertainty',
                severity: 'high',
                description: 'Target variable based on RFM patterns, not actual defaults. Model predicts behavioral risk, not true credit default.',
                mitigation: 'Conservative thresholds, continuous monitoring, and validation against actual outcomes as data becomes available.',
              },
              {
                title: 'Limited Historical Data',
                severity: 'medium',
                description: 'Only 90 days of transaction history available. Model may not capture long-term patterns or seasonal variations.',
                mitigation: 'Temporal validation, model recalibration as more data becomes available, and conservative risk thresholds.',
              },
              {
                title: 'Data Quality Challenges',
                severity: 'medium',
                description: 'Training data contains 25% outliers and rare categories. Model performance may degrade on edge cases.',
                mitigation: 'Robust scaling, business validation of predictions, and manual review for unusual patterns.',
              },
              {
                title: 'External Validation Gap',
                severity: 'high',
                description: 'Cannot validate against true defaults initially. Model performance on actual defaults is unknown.',
                mitigation: 'Post-deployment monitoring, A/B testing, and model refinement based on actual outcomes.',
              },
              {
                title: 'Context Limitations',
                severity: 'low',
                description: 'Model does not consider external factors like economic conditions, market changes, or customer life events.',
                mitigation: 'Regular model updates, incorporation of external signals, and human oversight for significant decisions.',
              },
            ].map((limitation, idx) => (
              <div
                key={idx}
                className="bg-white rounded-lg p-6 border-2 border-slate-200 hover:border-blue-300 transition-all"
              >
                <div className="flex items-start justify-between mb-3">
                  <h4 className="text-lg font-bold text-slate-800">{limitation.title}</h4>
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    limitation.severity === 'high' ? 'bg-red-100 text-red-700' :
                    limitation.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-blue-100 text-blue-700'
                  }`}>
                    {limitation.severity.toUpperCase()} SEVERITY
                  </span>
                </div>
                <p className="text-slate-700 mb-3">{limitation.description}</p>
                <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                  <div className="text-sm font-semibold text-slate-800 mb-1">Mitigation:</div>
                  <p className="text-sm text-slate-700">{limitation.mitigation}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Compliance Tab */}
      {activeTab === 'compliance' && (
        <div className="space-y-6 animate-fade-in">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="w-6 h-6 text-blue-600" />
                <h3 className="text-lg font-bold text-slate-800">OSFI Guideline E-23</h3>
              </div>
              <p className="text-slate-700 mb-4 text-sm">
                Office of the Superintendent of Financial Institutions (Canada) - Model Risk Management
              </p>
              <ul className="space-y-2 text-sm text-slate-700">
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Comprehensive model documentation</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Model validation and testing procedures</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Ongoing monitoring and governance</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Bias and fairness assessment</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border border-purple-200">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="w-6 h-6 text-purple-600" />
                <h3 className="text-lg font-bold text-slate-800">EU AI Act</h3>
              </div>
              <p className="text-slate-700 mb-4 text-sm">
                European Union Artificial Intelligence Act - High-Risk AI System Requirements
              </p>
              <ul className="space-y-2 text-sm text-slate-700">
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Transparency and explainability (SHAP explanations)</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Human oversight and monitoring</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Accuracy and robustness requirements</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span>Adverse action notifications (CFPB compliance)</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Compliance Status</h3>
            <div className="space-y-3">
              {[
                { requirement: 'Model Documentation', status: 'Compliant', framework: 'OSFI E-23, EU AI Act' },
                { requirement: 'Bias Assessment', status: 'Compliant', framework: 'OSFI E-23, EU AI Act' },
                { requirement: 'Explainability', status: 'Compliant', framework: 'EU AI Act' },
                { requirement: 'Human Oversight', status: 'Compliant', framework: 'EU AI Act' },
                { requirement: 'Adverse Action Notifications', status: 'Compliant', framework: 'CFPB' },
                { requirement: 'Model Monitoring', status: 'Compliant', framework: 'OSFI E-23' },
              ].map((item, idx) => (
                <div key={idx} className="flex items-center justify-between bg-white rounded-lg p-4 border border-slate-200">
                  <div>
                    <div className="font-semibold text-slate-800">{item.requirement}</div>
                    <div className="text-sm text-slate-600">{item.framework}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <span className="font-semibold text-green-600">{item.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelCard;
