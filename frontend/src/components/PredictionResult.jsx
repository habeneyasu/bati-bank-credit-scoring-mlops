import React from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, XCircle, Info } from 'lucide-react';
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';

const PredictionResult = ({ prediction }) => {
  const { risk_level, probability, prediction: pred } = prediction;

  const getRiskConfig = () => {
    switch (risk_level) {
      case 'low':
        return {
          color: 'green',
          icon: CheckCircle,
          gradient: 'from-green-500 to-emerald-500',
          bgGradient: 'from-green-50 to-emerald-50',
          text: 'Low Risk',
          recommendation: 'Auto-approve. Customer shows low risk indicators.',
        };
      case 'medium':
        return {
          color: 'yellow',
          icon: AlertTriangle,
          gradient: 'from-yellow-500 to-amber-500',
          bgGradient: 'from-yellow-50 to-amber-50',
          text: 'Medium Risk',
          recommendation: 'Manual review required. Additional verification may be needed.',
        };
      case 'high':
        return {
          color: 'red',
          icon: XCircle,
          gradient: 'from-red-500 to-rose-500',
          bgGradient: 'from-red-50 to-rose-50',
          text: 'High Risk',
          recommendation: 'Auto-reject. Customer shows high risk indicators.',
        };
      default:
        return {
          color: 'gray',
          icon: Info,
          gradient: 'from-gray-500 to-slate-500',
          bgGradient: 'from-gray-50 to-slate-50',
          text: 'Unknown',
          recommendation: 'Unable to determine risk level.',
        };
    }
  };

  const config = getRiskConfig();
  const Icon = config.icon;
  const riskPercentage = (probability * 100).toFixed(2);
  const safePercentage = ((1 - probability) * 100).toFixed(2);

  const pieData = [
    { name: 'Risk Probability', value: parseFloat(riskPercentage), fill: `url(#riskGradient)` },
    { name: 'Safe Probability', value: parseFloat(safePercentage), fill: '#e2e8f0' },
  ];

  return (
    <div className="card animate-fade-in">
      <div className={`bg-gradient-to-br ${config.bgGradient} rounded-xl p-6 mb-6 border-2 border-${config.color}-200`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`p-3 bg-white rounded-full shadow-lg`}>
              <Icon className={`w-8 h-8 text-${config.color}-600`} />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-slate-800">Risk Assessment</h3>
              <p className="text-slate-600">Prediction Result</p>
            </div>
          </div>
          <div className={`risk-badge ${risk_level} text-lg px-6 py-3`}>
            {config.text}
          </div>
        </div>

        {/* Probability Visualization */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Pie Chart */}
          <div className="bg-white rounded-lg p-4 shadow-md">
            <h4 className="text-sm font-semibold text-slate-600 mb-4 text-center">
              Risk Distribution
            </h4>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <defs>
                  <linearGradient id="riskGradient" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor={risk_level === 'low' ? '#10b981' : risk_level === 'medium' ? '#f59e0b' : '#ef4444'} />
                    <stop offset="100%" stopColor={risk_level === 'low' ? '#059669' : risk_level === 'medium' ? '#d97706' : '#dc2626'} />
                  </linearGradient>
                </defs>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  startAngle={90}
                  endAngle={-270}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => `${value}%`}
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    padding: '8px',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="text-center mt-2">
              <span className="text-2xl font-bold text-slate-800">{riskPercentage}%</span>
              <span className="text-sm text-slate-500 ml-1">Risk Probability</span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="bg-white rounded-lg p-4 shadow-md">
            <h4 className="text-sm font-semibold text-slate-600 mb-4">
              Risk Level Breakdown
            </h4>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-600">Low Risk</span>
                  <span className="font-semibold text-green-600">
                    {probability < 0.3 ? (probability * 100).toFixed(1) : 0}%
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-green-500 to-emerald-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min((probability < 0.3 ? probability : 0) * 100, 100)}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-600">Medium Risk</span>
                  <span className="font-semibold text-yellow-600">
                    {probability >= 0.3 && probability <= 0.6 ? (probability * 100).toFixed(1) : 0}%
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-yellow-500 to-amber-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min((probability >= 0.3 && probability <= 0.6 ? probability : 0) * 100, 100)}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-600">High Risk</span>
                  <span className="font-semibold text-red-600">
                    {probability > 0.6 ? (probability * 100).toFixed(1) : 0}%
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-red-500 to-rose-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min((probability > 0.6 ? probability : 0) * 100, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Details */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="text-sm text-slate-600 mb-1">Prediction</div>
          <div className="text-2xl font-bold text-slate-800">
            {pred === 1 ? (
              <span className="text-red-600 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                High Risk
              </span>
            ) : (
              <span className="text-green-600 flex items-center gap-2">
                <TrendingDown className="w-5 h-5" />
                Low Risk
              </span>
            )}
          </div>
        </div>
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="text-sm text-slate-600 mb-1">Risk Probability</div>
          <div className="text-2xl font-bold text-slate-800">{riskPercentage}%</div>
        </div>
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="text-sm text-slate-600 mb-1">Risk Level</div>
          <div className={`text-2xl font-bold ${
            risk_level === 'low' ? 'text-green-600' :
            risk_level === 'medium' ? 'text-yellow-600' : 'text-red-600'
          }`}>{config.text}</div>
        </div>
      </div>

      {/* Recommendation */}
      <div className={`bg-gradient-to-r ${config.bgGradient} rounded-lg p-6 border-l-4 ${
        risk_level === 'low' ? 'border-green-500' :
        risk_level === 'medium' ? 'border-yellow-500' : 'border-red-500'
      }`}>
        <div className="flex items-start gap-3">
          <Icon className={`w-6 h-6 flex-shrink-0 mt-1 ${
            risk_level === 'low' ? 'text-green-600' :
            risk_level === 'medium' ? 'text-yellow-600' : 'text-red-600'
          }`} />
          <div>
            <h4 className="font-bold text-slate-800 mb-2">Recommendation</h4>
            <p className="text-slate-700 leading-relaxed">{config.recommendation}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
