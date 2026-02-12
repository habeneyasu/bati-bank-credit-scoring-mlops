import React from 'react';
import { Brain, TrendingUp, TrendingDown, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ExplanationPanel = ({ explanation }) => {
  const topFeatures = explanation.feature_importance?.slice(0, 15) || [];
  
  // Prepare data for bar chart
  const chartData = topFeatures.map((feat, idx) => ({
    name: feat.feature.length > 20 ? feat.feature.substring(0, 20) + '...' : feat.feature,
    fullName: feat.feature,
    value: feat.shap_value,
    absValue: Math.abs(feat.shap_value),
    featureValue: feat.feature_value,
    isPositive: feat.shap_value > 0,
  })).sort((a, b) => b.absValue - a.absValue);

  const maxAbsValue = Math.max(...chartData.map(d => d.absValue));

  return (
    <div className="card animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-indigo-100 rounded-lg">
          <Brain className="w-6 h-6 text-indigo-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Model Explanation</h2>
          <p className="text-slate-600 text-sm">SHAP-based feature importance analysis</p>
        </div>
      </div>

      {/* Summary */}
      {explanation.explanation_summary && (
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg p-4 mb-6 border-l-4 border-indigo-500">
          <div className="flex items-start gap-2">
            <Info className="w-5 h-5 text-indigo-600 flex-shrink-0 mt-0.5" />
            <p className="text-slate-700 text-sm leading-relaxed">{explanation.explanation_summary}</p>
          </div>
        </div>
      )}

      {/* Base Value */}
      <div className="bg-slate-50 rounded-lg p-4 mb-6 border border-slate-200">
        <div className="flex justify-between items-center">
          <span className="text-slate-600 font-medium">Base Value (Expected Risk)</span>
          <span className="text-2xl font-bold text-slate-800">
            {explanation.base_value?.toFixed(4) || 'N/A'}
          </span>
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="mb-6">
        <h3 className="text-lg font-bold text-slate-800 mb-4">Top Contributing Features</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 200, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number"
              domain={[-maxAbsValue, maxAbsValue]}
              tick={{ fill: '#64748b', fontSize: 12 }}
              stroke="#cbd5e1"
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: '#64748b', fontSize: 11 }}
              stroke="#cbd5e1"
              width={180}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-3 rounded-lg shadow-lg border border-slate-200">
                      <p className="font-semibold text-slate-800 mb-2">{data.fullName}</p>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between gap-4">
                          <span className="text-slate-600">SHAP Value:</span>
                          <span className={`font-bold ${data.isPositive ? 'text-red-600' : 'text-green-600'}`}>
                            {data.isPositive ? '+' : ''}{data.value.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex justify-between gap-4">
                          <span className="text-slate-600">Feature Value:</span>
                          <span className="font-semibold text-slate-800">{data.featureValue.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between gap-4">
                          <span className="text-slate-600">Impact:</span>
                          <span className={`font-semibold ${data.isPositive ? 'text-red-600' : 'text-green-600'}`}>
                            {data.isPositive ? 'Increases Risk' : 'Decreases Risk'}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="value" radius={[0, 8, 8, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.isPositive ? '#ef4444' : '#10b981'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature List */}
      <div className="space-y-3">
        <h3 className="text-lg font-bold text-slate-800 mb-4">Detailed Feature Impact</h3>
        <div className="max-h-96 overflow-y-auto space-y-2 pr-2">
          {topFeatures.map((feat, idx) => {
            const isPositive = feat.shap_value > 0;
            const absValue = Math.abs(feat.shap_value);
            const percentage = (absValue / maxAbsValue) * 100;

            return (
              <div
                key={idx}
                className="bg-slate-50 rounded-lg p-4 border border-slate-200 hover:border-indigo-300 transition-all"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {isPositive ? (
                      <TrendingUp className="w-4 h-4 text-red-500" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-green-500" />
                    )}
                    <span className="font-semibold text-slate-800">{feat.feature}</span>
                  </div>
                  <span
                    className={`font-bold ${
                      isPositive ? 'text-red-600' : 'text-green-600'
                    }`}
                  >
                    {isPositive ? '+' : ''}{feat.shap_value.toFixed(4)}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-1 bg-slate-200 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        isPositive
                          ? 'bg-gradient-to-r from-red-500 to-rose-500'
                          : 'bg-gradient-to-r from-green-500 to-emerald-500'
                      }`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-500 w-24 text-right">
                    Value: {feat.feature_value.toFixed(4)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ExplanationPanel;
