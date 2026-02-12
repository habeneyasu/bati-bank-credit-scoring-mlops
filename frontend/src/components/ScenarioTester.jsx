import React, { useState } from 'react';
import { Sliders, RefreshCw, ArrowRight, TrendingUp, TrendingDown } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const ScenarioTester = ({ initialFeatures, featureNames, currentPrediction }) => {
  const [scenarioFeatures, setScenarioFeatures] = useState({ ...initialFeatures });
  const [scenarioResult, setScenarioResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState([]);

  // Get top 5 features from current explanation
  const topFeatures = currentPrediction?.explanation?.feature_importance?.slice(0, 5) || [];

  React.useEffect(() => {
    if (topFeatures.length > 0) {
      const featureIndices = topFeatures.map(f => featureNames.indexOf(f.feature)).filter(idx => idx >= 0);
      setSelectedFeatures(featureIndices);
      
      // Initialize scenario features with current values
      const initial = { ...initialFeatures };
      topFeatures.forEach(f => {
        const idx = featureNames.indexOf(f.feature);
        if (idx >= 0) {
          initial[idx] = f.feature_value;
        }
      });
      setScenarioFeatures(initial);
    }
  }, [topFeatures, featureNames, initialFeatures]);

  const handleFeatureChange = (idx, value) => {
    setScenarioFeatures({
      ...scenarioFeatures,
      [idx]: parseFloat(value) || 0,
    });
  };

  const testScenario = async () => {
    setLoading(true);
    try {
      const featureArray = Object.values(scenarioFeatures);
      const result = await creditScoringAPI.predict(featureArray, true);
      setScenarioResult(result);
      
      // Add to history
      setHistory(prev => [
        ...prev,
        {
          features: { ...scenarioFeatures },
          prediction: result,
          timestamp: new Date(),
        },
      ]);
    } catch (error) {
      alert('Scenario test failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const resetScenario = () => {
    setScenarioFeatures({ ...initialFeatures });
    setScenarioResult(null);
  };

  const historyData = history.map((item, idx) => ({
    scenario: `S${idx + 1}`,
    probability: item.prediction.probability * 100,
    risk: item.prediction.risk_level,
  }));

  return (
    <div className="card animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-purple-100 rounded-lg">
          <Sliders className="w-6 h-6 text-purple-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Scenario Testing</h2>
          <p className="text-slate-600 text-sm">Adjust features to see how they affect risk prediction</p>
        </div>
      </div>

      {/* Feature Adjustments */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Adjust Key Features</h3>
        <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
          {topFeatures.map((feat, idx) => {
            const featureIdx = featureNames.indexOf(feat.feature);
            if (featureIdx < 0) return null;
            
            const currentValue = scenarioFeatures[featureIdx] || 0;
            const originalValue = feat.feature_value;
            const difference = currentValue - originalValue;

            return (
              <div key={idx} className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <label className="font-semibold text-slate-800 block">{feat.feature}</label>
                    <div className="text-xs text-slate-500 mt-1">
                      Original: {originalValue.toFixed(4)}
                      {difference !== 0 && (
                        <span className={`ml-2 ${difference > 0 ? 'text-red-600' : 'text-green-600'}`}>
                          ({difference > 0 ? '+' : ''}{difference.toFixed(4)})
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min={originalValue - 2}
                    max={originalValue + 2}
                    step="0.01"
                    value={currentValue}
                    onChange={(e) => handleFeatureChange(featureIdx, e.target.value)}
                    className="flex-1"
                  />
                  <input
                    type="number"
                    step="0.0001"
                    value={currentValue}
                    onChange={(e) => handleFeatureChange(featureIdx, e.target.value)}
                    className="w-24 input-field text-sm"
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={testScenario}
          disabled={loading}
          className="btn-primary flex-1 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Testing...
            </>
          ) : (
            <>
              <Sliders className="w-4 h-4" />
              Test Scenario
            </>
          )}
        </button>
        <button
          onClick={resetScenario}
          className="btn-secondary flex items-center justify-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reset
        </button>
      </div>

      {/* Scenario Result */}
      {scenarioResult && (
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg p-6 border-2 border-purple-200 mb-6">
          <h3 className="text-lg font-bold text-slate-800 mb-4">Scenario Result</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="text-sm text-slate-600 mb-1">Previous Risk</div>
              <div className="text-2xl font-bold text-slate-800 capitalize">
                {currentPrediction.risk_level}
              </div>
              <div className="text-sm text-slate-500 mt-1">
                {(currentPrediction.probability * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="text-sm text-slate-600 mb-1">New Risk</div>
              <div className={`text-2xl font-bold capitalize ${
                scenarioResult.risk_level === 'low' ? 'text-green-600' :
                scenarioResult.risk_level === 'medium' ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {scenarioResult.risk_level}
              </div>
              <div className="text-sm text-slate-500 mt-1">
                {(scenarioResult.probability * 100).toFixed(2)}%
              </div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-purple-200">
            <div className="flex items-center gap-2">
              {scenarioResult.probability > currentPrediction.probability ? (
                <TrendingUp className="w-5 h-5 text-red-600" />
              ) : (
                <TrendingDown className="w-5 h-5 text-green-600" />
              )}
              <span className="text-slate-700 font-semibold">
                Probability Change: {((scenarioResult.probability - currentPrediction.probability) * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* History Chart */}
      {history.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Scenario History</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={historyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="scenario" tick={{ fill: '#64748b', fontSize: 12 }} stroke="#cbd5e1" />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#64748b', fontSize: 12 }}
                stroke="#cbd5e1"
                label={{ value: 'Risk %', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                formatter={(value) => `${value.toFixed(2)}%`}
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
              />
              <Line
                type="monotone"
                dataKey="probability"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ fill: '#3b82f6', r: 4 }}
                name="Risk Probability"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default ScenarioTester;
