import React, { useState, useEffect } from 'react';
import { TrendingUp, Brain, BarChart3, Zap, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import FeatureInputForm from '../components/FeatureInputForm';
import PredictionResult from '../components/PredictionResult';
import ExplanationPanel from '../components/ExplanationPanel';
import ScenarioTester from '../components/ScenarioTester';
import { creditScoringAPI } from '../utils/api';

const Dashboard = () => {
  const [featureNames, setFeatureNames] = useState([]);
  const [features, setFeatures] = useState({});
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [activeTab, setActiveTab] = useState('predict');
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    loadFeatureNames();
    checkApiHealth();
  }, []);

  const loadFeatureNames = async () => {
    try {
      const data = await creditScoringAPI.getFeatureNames();
      setFeatureNames(data.feature_names || []);
      
      // Initialize features with zeros
      const initialFeatures = {};
      data.feature_names?.forEach((name, idx) => {
        initialFeatures[idx] = 0;
      });
      setFeatures(initialFeatures);
    } catch (error) {
      console.error('Failed to load feature names:', error);
      // Use default feature names
      const defaultNames = Array.from({ length: 26 }, (_, i) => `feature_${i}`);
      setFeatureNames(defaultNames);
      const initialFeatures = {};
      defaultNames.forEach((_, idx) => {
        initialFeatures[idx] = 0;
      });
      setFeatures(initialFeatures);
    }
  };

  const checkApiHealth = async () => {
    try {
      const health = await creditScoringAPI.healthCheck();
      setApiStatus(health.model_loaded ? 'healthy' : 'degraded');
    } catch (error) {
      setApiStatus('error');
    }
  };

  const loadSampleData = () => {
    const sampleFeatures = [
      0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994,
      -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ];
    
    const newFeatures = {};
    sampleFeatures.forEach((value, idx) => {
      newFeatures[idx] = value;
    });
    setFeatures(newFeatures);
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const featureArray = Object.values(features);
      const result = await creditScoringAPI.predict(featureArray, true);
      setPrediction(result);
      if (result.explanation) {
        setExplanation(result.explanation);
      }
      setActiveTab('result');
    } catch (error) {
      alert('Prediction failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    setLoading(true);
    try {
      const featureArray = Object.values(features);
      const result = await creditScoringAPI.explain(featureArray);
      setExplanation(result);
      setActiveTab('explanation');
    } catch (error) {
      alert('Explanation failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = () => {
    switch (apiStatus) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'degraded':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      default:
        return <XCircle className="w-5 h-5 text-red-500" />;
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-2xl">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
                <TrendingUp className="w-10 h-10" />
                Credit Risk Scoring Dashboard
              </h1>
              <p className="text-blue-100 text-lg">
                Interactive tool for loan officers and credit analysts
              </p>
            </div>
            <div className="flex items-center gap-2 bg-white/20 backdrop-blur-sm px-4 py-2 rounded-lg">
              {getStatusIcon()}
              <span className="font-semibold">
                {apiStatus === 'healthy' ? 'System Online' : 
                 apiStatus === 'degraded' ? 'Degraded' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Feature Input */}
          <div className="lg:col-span-1">
            <div className="card sticky top-6">
              <div className="flex items-center gap-2 mb-6">
                <Brain className="w-6 h-6 text-indigo-600" />
                <h2 className="text-2xl font-bold text-slate-800">Customer Features</h2>
              </div>
              
              <FeatureInputForm
                featureNames={featureNames}
                features={features}
                onChange={setFeatures}
              />

              <div className="mt-6 space-y-3">
                <button
                  onClick={loadSampleData}
                  className="btn-secondary w-full flex items-center justify-center gap-2"
                >
                  <Zap className="w-4 h-4" />
                  Load Sample Data
                </button>
                
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={handlePredict}
                    disabled={loading}
                    className="btn-primary flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <BarChart3 className="w-4 h-4" />
                        Predict
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={handleExplain}
                    disabled={loading}
                    className="btn-secondary flex items-center justify-center gap-2"
                  >
                    <Brain className="w-4 h-4" />
                    Explain
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Tab Navigation */}
            {prediction && (
              <div className="card">
                <div className="flex gap-2 border-b border-slate-200">
                  <button
                    onClick={() => setActiveTab('result')}
                    className={`px-4 py-2 font-semibold transition-colors ${
                      activeTab === 'result'
                        ? 'text-blue-600 border-b-2 border-blue-600'
                        : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    Prediction Result
                  </button>
                  {explanation && (
                    <button
                      onClick={() => setActiveTab('explanation')}
                      className={`px-4 py-2 font-semibold transition-colors ${
                        activeTab === 'explanation'
                          ? 'text-blue-600 border-b-2 border-blue-600'
                          : 'text-slate-500 hover:text-slate-700'
                      }`}
                    >
                      Explanation
                    </button>
                  )}
                  <button
                    onClick={() => setActiveTab('scenario')}
                    className={`px-4 py-2 font-semibold transition-colors ${
                      activeTab === 'scenario'
                        ? 'text-blue-600 border-b-2 border-blue-600'
                        : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    Scenario Testing
                  </button>
                </div>
              </div>
            )}

            {/* Prediction Result */}
            {activeTab === 'result' && prediction && (
              <PredictionResult prediction={prediction} />
            )}

            {/* Explanation */}
            {activeTab === 'explanation' && explanation && (
              <ExplanationPanel explanation={explanation} />
            )}

            {/* Scenario Tester */}
            {activeTab === 'scenario' && prediction && explanation && (
              <ScenarioTester
                initialFeatures={features}
                featureNames={featureNames}
                currentPrediction={prediction}
              />
            )}

            {/* Empty State */}
            {!prediction && (
              <div className="card text-center py-16">
                <BarChart3 className="w-16 h-16 text-slate-300 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-slate-600 mb-2">
                  No Prediction Yet
                </h3>
                <p className="text-slate-500">
                  Enter customer features and click "Predict" to get started
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
