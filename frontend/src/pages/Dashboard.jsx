import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, Brain, BarChart3, Zap, AlertCircle, CheckCircle, XCircle, 
  FileText, Shield, X, GitBranch, Activity, Home, Settings, 
  Users, Target, Clock, Award, ArrowRight, Menu, X as XIcon, KeyRound, LogOut, User as UserIcon, Database, List, FileCheck, FlaskConical, RefreshCw
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import FeatureInputForm from '../components/FeatureInputForm';
import PredictionResult from '../components/PredictionResult';
import ExplanationPanel from '../components/ExplanationPanel';
import ScenarioTester from '../components/ScenarioTester';
import ModelCard from '../components/ModelCard';
import FairnessAnalysis from '../components/FairnessAnalysis';
import VersionInfo from '../components/VersionInfo';
import PerformanceMonitor from '../components/PerformanceMonitor';
import PredictionsTable from '../components/PredictionsTable';
import BusinessKPIs from '../components/BusinessKPIs';
import UsersTable from '../components/UsersTable';
import RolesTable from '../components/RolesTable';
import DataUpload from '../components/DataUpload';
import TransactionsTable from '../components/TransactionsTable';
import DriftDetection from '../components/DriftDetection';
import AlertsPanel from '../components/AlertsPanel';
import DataQualityMonitor from '../components/DataQualityMonitor';
import CustomerScoresTable from '../components/CustomerScoresTable';
import CustomerScorer from '../components/CustomerScorer';
import DataLineage from '../components/DataLineage';
import FeatureStore from '../components/FeatureStore';
import ABTesting from '../components/ABTesting';
import ModelRetraining from '../components/ModelRetraining';
import BatchPredictions from '../components/BatchPredictions';
import LoadTesting from '../components/LoadTesting';
import { creditScoringAPI } from '../utils/api';

const Dashboard = () => {
  const [featureNames, setFeatureNames] = useState([]);
  const [features, setFeatures] = useState({});
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [resultTab, setResultTab] = useState('result');
  const [apiStatus, setApiStatus] = useState('checking');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [performanceData, setPerformanceData] = useState(null);
  const [featureStoreStats, setFeatureStoreStats] = useState(null);

  useEffect(() => {
    loadFeatureNames();
    checkApiHealth();
    loadPerformanceData();
    loadFeatureStoreStats();
  }, []);

  const loadFeatureNames = async () => {
    try {
      const data = await creditScoringAPI.getFeatureNames();
      setFeatureNames(data.feature_names || []);
      
      const initialFeatures = {};
      data.feature_names?.forEach((name, idx) => {
        initialFeatures[idx] = 0;
      });
      setFeatures(initialFeatures);
    } catch (error) {
      console.error('Failed to load feature names:', error);
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

  const loadPerformanceData = async () => {
    try {
      const data = await creditScoringAPI.getPerformanceMetrics();
      setPerformanceData(data);
    } catch (error) {
      console.error('Failed to load performance data:', error);
    }
  };

  const loadFeatureStoreStats = async () => {
    try {
      const data = await creditScoringAPI.getFeatureStoreStats();
      setFeatureStoreStats(data);
    } catch (error) {
      console.error('Failed to load feature store stats:', error);
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
      setActiveTab('predict');
      setResultTab('result');
      loadPerformanceData(); // Refresh performance data
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
      setActiveTab('predict');
      setResultTab('explanation');
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

  const menuItems = [
    { id: 'overview', label: 'Overview', icon: Home },
    { id: 'data', label: 'Data Upload', icon: Database },
    { id: 'transactions', label: 'Transactions', icon: List },
    { id: 'predict', label: 'Risk Assessment', icon: Brain },
    { id: 'predictions', label: 'Predictions', icon: FileText },
    { id: 'scores', label: 'Customer Scores', icon: Award },
    { id: 'score-customer', label: 'Score Customer', icon: Award },
    { id: 'feature-store', label: 'Feature Store', icon: Database },
    { id: 'ab-testing', label: 'A/B Testing', icon: FlaskConical },
    { id: 'retraining', label: 'Model Retraining', icon: RefreshCw },
    { id: 'batch-predictions', label: 'Batch Predictions', icon: FileText },
    { id: 'load-testing', label: 'Load Testing', icon: Zap },
    { id: 'kpis', label: 'Business KPIs', icon: BarChart3 },
    { id: 'drift', label: 'Drift Detection', icon: TrendingUp },
    { id: 'alerts', label: 'Alerts', icon: AlertCircle },
    { id: 'data-quality', label: 'Data Quality', icon: FileCheck },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'roles', label: 'Roles & Permissions', icon: KeyRound },
    { id: 'performance', label: 'Performance', icon: Activity },
    { id: 'governance', label: 'Governance', icon: Shield },
    { id: 'versions', label: 'Versions', icon: GitBranch },
    { id: 'lineage', label: 'Data Lineage', icon: GitBranch },
  ];

  const stats = performanceData?.stats?.all || {};
  const p95Latency = stats.p95 || 0;
  const totalRequests = stats.count || 0;
  const errorRate = performanceData?.stats?.error_rate?.rate || 0;

  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Top Navigation Bar */}
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <Menu className="w-6 h-6 text-slate-600" />
              </button>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-slate-800">Credit Risk Scoring</h1>
                  <p className="text-xs text-slate-500">ML-Powered Decision Platform</p>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-2 bg-slate-50 px-4 py-2 rounded-lg">
                {getStatusIcon()}
                <span className="text-sm font-medium text-slate-700">
                  {apiStatus === 'healthy' ? 'System Online' : 
                   apiStatus === 'degraded' ? 'Degraded' : 'Offline'}
                </span>
              </div>
              {user && (
                <div className="flex items-center gap-3">
                  <div className="text-right hidden sm:block">
                    <div className="text-sm font-medium text-slate-800">{user.full_name || user.username}</div>
                    <div className="text-xs text-slate-500">{user.department || 'User'}</div>
                  </div>
                  <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                    <UserIcon className="w-5 h-5 text-blue-600" />
                  </div>
                  <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100 rounded-lg transition"
                    title="Logout"
                  >
                    <LogOut className="w-4 h-4" />
                    <span className="hidden sm:inline">Logout</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar Navigation */}
        <aside className={`
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          fixed lg:static lg:translate-x-0
          w-64 bg-white border-r border-slate-200 h-[calc(100vh-73px)]
          transition-transform duration-300 z-40
          overflow-y-auto
        `}>
          <div className="p-4">
            <div className="flex items-center justify-between mb-6 lg:hidden">
              <h2 className="text-lg font-bold text-slate-800">Menu</h2>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 hover:bg-slate-100 rounded-lg"
              >
                <XIcon className="w-5 h-5 text-slate-600" />
              </button>
            </div>
            <nav className="space-y-1">
              {menuItems.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActiveTab(item.id);
                      if (window.innerWidth < 1024) setSidebarOpen(false);
                    }}
                    className={`
                      w-full flex items-center gap-3 px-4 py-3 rounded-lg
                      transition-all duration-200 text-left
                      ${activeTab === item.id
                        ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md'
                        : 'text-slate-700 hover:bg-slate-100'
                      }
                    `}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{item.label}</span>
                    {activeTab === item.id && (
                      <ArrowRight className="w-4 h-4 ml-auto" />
                    )}
                  </button>
                );
              })}
            </nav>
          </div>
        </aside>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div
            className="lg:hidden fixed inset-0 bg-black/20 z-30"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 p-6 lg:p-8">
          {/* Overview Section */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Hero Section with KPIs */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
                <div className="card bg-gradient-to-br from-blue-600 to-indigo-600 text-white">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-3 bg-white/20 rounded-lg">
                      <Target className="w-6 h-6" />
                    </div>
                    <span className="text-sm opacity-90">Total Requests</span>
                  </div>
                  <div className="text-3xl font-bold mb-1">{totalRequests.toLocaleString()}</div>
                  <div className="text-sm opacity-75">Predictions processed</div>
                </div>

                <div className="card bg-gradient-to-br from-green-500 to-emerald-500 text-white">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-3 bg-white/20 rounded-lg">
                      <Clock className="w-6 h-6" />
                    </div>
                    <span className="text-sm opacity-90">P95 Latency</span>
                  </div>
                  <div className="text-3xl font-bold mb-1">
                    {p95Latency > 0 ? `${p95Latency.toFixed(1)}ms` : 'N/A'}
                  </div>
                  <div className="text-sm opacity-75">
                    {p95Latency > 0 && p95Latency < 200 ? '✓ SLA Compliant' : 'No data yet'}
                  </div>
                </div>

                <div className="card bg-gradient-to-br from-purple-500 to-pink-500 text-white">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-3 bg-white/20 rounded-lg">
                      <Award className="w-6 h-6" />
                    </div>
                    <span className="text-sm opacity-90">Error Rate</span>
                  </div>
                  <div className="text-3xl font-bold mb-1">
                    {(errorRate * 100).toFixed(2)}%
                  </div>
                  <div className="text-sm opacity-75">System reliability</div>
                </div>

                <div className="card bg-gradient-to-br from-amber-500 to-orange-500 text-white">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-3 bg-white/20 rounded-lg">
                      <Users className="w-6 h-6" />
                    </div>
                    <span className="text-sm opacity-90">Model Status</span>
                  </div>
                  <div className="text-3xl font-bold mb-1">
                    {apiStatus === 'healthy' ? 'Active' : 'Degraded'}
                  </div>
                  <div className="text-sm opacity-75">Production ready</div>
                </div>

                <div className="card bg-gradient-to-br from-cyan-500 to-teal-500 text-white">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-3 bg-white/20 rounded-lg">
                      <Database className="w-6 h-6" />
                    </div>
                    <span className="text-sm opacity-90">Feature Store</span>
                  </div>
                  <div className="text-3xl font-bold mb-1">
                    {featureStoreStats?.total_features?.toLocaleString() || 0}
                  </div>
                  <div className="text-sm opacity-75">
                    {featureStoreStats?.cache_coverage 
                      ? `${featureStoreStats.cache_coverage.toFixed(1)}% coverage`
                      : 'Cached features'}
                  </div>
                </div>
              </div>

              {/* Welcome Section */}
              <div className="card bg-gradient-to-br from-white to-blue-50 border-2 border-blue-200">
                <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                  <div className="flex-1">
                    <h2 className="text-3xl font-bold text-slate-800 mb-3">
                      Welcome to Credit Risk Scoring Platform
                    </h2>
                    <p className="text-slate-600 text-lg mb-4">
                      AI-powered credit risk assessment for real-time lending decisions. 
                      Built with industry-leading ML models and regulatory compliance.
                    </p>
                    <div className="flex flex-wrap gap-3">
                      <div className="flex items-center gap-2 text-sm text-slate-600">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>Basel II Compliant</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-slate-600">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>Sub-200ms Latency</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-slate-600">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>SHAP Explanations</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-slate-600">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>Fairness Monitoring</span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => setActiveTab('predict')}
                    className="btn-primary flex items-center gap-2 px-8 py-4 text-lg"
                  >
                    Start Assessment
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="card hover:shadow-xl transition-all cursor-pointer" onClick={() => setActiveTab('predict')}>
                  <div className="flex items-center gap-4">
                    <div className="p-4 bg-blue-100 rounded-lg">
                      <Brain className="w-8 h-8 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-800">Risk Assessment</h3>
                      <p className="text-sm text-slate-600">Evaluate customer credit risk</p>
                    </div>
                  </div>
                </div>

                <div className="card hover:shadow-xl transition-all cursor-pointer" onClick={() => setActiveTab('performance')}>
                  <div className="flex items-center gap-4">
                    <div className="p-4 bg-green-100 rounded-lg">
                      <Activity className="w-8 h-8 text-green-600" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-800">Performance</h3>
                      <p className="text-sm text-slate-600">Monitor system metrics</p>
                    </div>
                  </div>
                </div>

                <div className="card hover:shadow-xl transition-all cursor-pointer" onClick={() => setActiveTab('governance')}>
                  <div className="flex items-center gap-4">
                    <div className="p-4 bg-purple-100 rounded-lg">
                      <Shield className="w-8 h-8 text-purple-600" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-800">Governance</h3>
                      <p className="text-sm text-slate-600">Model compliance & fairness</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Data Upload Section */}
          {activeTab === 'data' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Data Upload</h2>
                  <p className="text-slate-600 mt-1">Upload raw transaction data to the database</p>
                </div>
              </div>
              <DataUpload />
            </div>
          )}

          {/* Transactions View Section */}
          {activeTab === 'transactions' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Transaction Data</h2>
                  <p className="text-slate-600 mt-1">View and filter uploaded transaction data from the database</p>
                </div>
              </div>
              <TransactionsTable />
            </div>
          )}

          {/* Risk Assessment Section */}
          {activeTab === 'predict' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Risk Assessment</h2>
                  <p className="text-slate-600 mt-1">Evaluate customer credit risk in real-time</p>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Feature Input */}
                <div className="lg:col-span-1">
                  <div className="card sticky top-24">
                    <div className="flex items-center gap-2 mb-6">
                      <Brain className="w-6 h-6 text-indigo-600" />
                      <h3 className="text-xl font-bold text-slate-800">Customer Features</h3>
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
                          onClick={() => setResultTab('result')}
                          className={`px-4 py-2 font-semibold transition-colors ${
                            resultTab === 'result'
                              ? 'text-blue-600 border-b-2 border-blue-600'
                              : 'text-slate-500 hover:text-slate-700'
                          }`}
                        >
                          Prediction Result
                        </button>
                        {explanation && (
                          <button
                            onClick={() => setResultTab('explanation')}
                            className={`px-4 py-2 font-semibold transition-colors ${
                              resultTab === 'explanation'
                                ? 'text-blue-600 border-b-2 border-blue-600'
                                : 'text-slate-500 hover:text-slate-700'
                            }`}
                          >
                            Explanation
                          </button>
                        )}
                        <button
                          onClick={() => setResultTab('scenario')}
                          className={`px-4 py-2 font-semibold transition-colors ${
                            resultTab === 'scenario'
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
                  {resultTab === 'result' && prediction && (
                    <PredictionResult prediction={prediction} />
                  )}

                  {/* Explanation */}
                  {resultTab === 'explanation' && explanation && (
                    <ExplanationPanel explanation={explanation} />
                  )}

                  {/* Scenario Tester */}
                  {resultTab === 'scenario' && prediction && explanation && (
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
          )}

          {/* Performance Section */}
          {activeTab === 'performance' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Performance Monitor</h2>
                  <p className="text-slate-600 mt-1">Real-time system metrics and SLA compliance</p>
                </div>
              </div>
              <PerformanceMonitor />
            </div>
          )}

          {/* Governance Section */}
          {activeTab === 'governance' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Model Governance</h2>
                  <p className="text-slate-600 mt-1">Compliance, fairness, and regulatory oversight</p>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-6">
                <ModelCard />
                <FairnessAnalysis />
              </div>
            </div>
          )}

          {/* Predictions Section */}
          {activeTab === 'predictions' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Predictions History</h2>
                  <p className="text-slate-600 mt-1">View all predictions stored in the database</p>
                </div>
              </div>
              <PredictionsTable />
            </div>
          )}

          {/* Customer Scores Section */}
          {activeTab === 'scores' && (
            <div className="space-y-6">
              <CustomerScoresTable />
            </div>
          )}

          {/* Score Customer Section */}
          {activeTab === 'score-customer' && (
            <div className="space-y-6">
              <CustomerScorer />
            </div>
          )}

          {/* Business KPIs Section */}
          {activeTab === 'kpis' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Business KPIs</h2>
                  <p className="text-slate-600 mt-1">Key performance indicators and analytics</p>
                </div>
              </div>
              <BusinessKPIs />
            </div>
          )}

          {/* Users Section */}
          {activeTab === 'users' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Users Management</h2>
                  <p className="text-slate-600 mt-1">Manage system users and their access</p>
                </div>
              </div>
              <UsersTable />
            </div>
          )}

          {/* Roles Section */}
          {activeTab === 'roles' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Roles & Permissions</h2>
                  <p className="text-slate-600 mt-1">Manage roles and their permissions</p>
                </div>
              </div>
              <RolesTable />
            </div>
          )}

          {/* Versions Section */}
          {activeTab === 'versions' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-slate-800">Version Information</h2>
                  <p className="text-slate-600 mt-1">Model and data version tracking</p>
                </div>
              </div>
              <VersionInfo />
            </div>
          )}

          {/* Drift Detection Section */}
          {activeTab === 'drift' && (
            <div className="space-y-6">
              <DriftDetection />
            </div>
          )}

          {/* Alerts Section */}
          {activeTab === 'alerts' && (
            <div className="space-y-6">
              <AlertsPanel />
            </div>
          )}

          {/* Data Lineage Section */}
          {activeTab === 'lineage' && (
            <div className="space-y-6">
              <DataLineage />
            </div>
          )}

          {/* Data Quality Section */}
          {activeTab === 'data-quality' && (
            <div className="space-y-6">
              <DataQualityMonitor />
            </div>
          )}

          {/* Feature Store Section */}
          {activeTab === 'feature-store' && (
            <div className="space-y-6">
              <FeatureStore />
            </div>
          )}

          {/* A/B Testing Section */}
          {activeTab === 'ab-testing' && (
            <div className="space-y-6">
              <ABTesting />
            </div>
          )}

          {/* Model Retraining Section */}
          {activeTab === 'retraining' && (
            <div className="space-y-6">
              <ModelRetraining />
            </div>
          )}

          {/* Batch Predictions Section */}
          {activeTab === 'batch-predictions' && (
            <div className="space-y-6">
              <BatchPredictions />
            </div>
          )}

          {/* Load Testing Section */}
          {activeTab === 'load-testing' && (
            <div className="space-y-6">
              <LoadTesting />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
