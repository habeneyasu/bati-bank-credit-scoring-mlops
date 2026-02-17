import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, Brain, BarChart3, Zap, AlertCircle, CheckCircle, XCircle, 
  FileText, Shield, X, GitBranch, Activity, Home, Settings, 
  Users, Target, Clock, Award, ArrowRight, Menu, X as XIcon, KeyRound, LogOut, User as UserIcon, 
  Database, List, FileCheck, FlaskConical, RefreshCw, Sparkles, TrendingDown, DollarSign,
  LineChart, PieChart, AlertTriangle, Bell, Eye, Lock, Globe, Server, Cpu, HardDrive,
  ChevronDown, ChevronUp
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
import ModelPerformanceValidation from '../components/ModelPerformanceValidation';
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
  const [kpiData, setKpiData] = useState(null);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [expandedCategories, setExpandedCategories] = useState({
    core: false,
    analytics: false,
    mlops: false,
    monitoring: false,
    governance: false,
  });

  // Organized menu items by category
  const menuCategories = [
    {
      id: 'core',
      label: 'Core Operations',
      items: [
        { id: 'overview', label: 'Overview', icon: Home },
        { id: 'predict', label: 'Risk Assessment', icon: Brain },
        { id: 'score-customer', label: 'Score Customer', icon: Target },
        { id: 'data', label: 'Data Upload', icon: Database },
        { id: 'transactions', label: 'Transactions', icon: List },
      ]
    },
    {
      id: 'analytics',
      label: 'Analytics & Insights',
      items: [
        { id: 'predictions', label: 'Predictions', icon: FileText },
        { id: 'scores', label: 'Customer Scores', icon: Award },
        { id: 'kpis', label: 'Business KPIs', icon: BarChart3 },
        { id: 'feature-store', label: 'Feature Store', icon: Database },
      ]
    },
    {
      id: 'mlops',
      label: 'ML Operations',
      items: [
        { id: 'ab-testing', label: 'A/B Testing', icon: FlaskConical },
        { id: 'retraining', label: 'Model Retraining', icon: RefreshCw },
        { id: 'batch-predictions', label: 'Batch Predictions', icon: FileText },
        { id: 'versions', label: 'Versions', icon: GitBranch },
      ]
    },
    {
      id: 'monitoring',
      label: 'Monitoring & Quality',
      items: [
        { id: 'drift', label: 'Drift Detection', icon: TrendingUp },
        { id: 'alerts', label: 'Alerts', icon: Bell },
        { id: 'data-quality', label: 'Data Quality', icon: FileCheck },
        { id: 'performance', label: 'Performance', icon: Activity },
        { id: 'model-validation', label: 'Model Validation', icon: BarChart3 },
      ]
    },
    {
      id: 'governance',
      label: 'Governance & Admin',
      items: [
        { id: 'governance', label: 'Governance', icon: Shield },
        { id: 'lineage', label: 'Data Lineage', icon: GitBranch },
        { id: 'users', label: 'Users', icon: Users },
        { id: 'roles', label: 'Roles & Permissions', icon: KeyRound },
        { id: 'load-testing', label: 'Load Testing', icon: Zap },
      ]
    },
  ];

  useEffect(() => {
    loadFeatureNames();
    checkApiHealth();
    loadPerformanceData();
    loadFeatureStoreStats();
    loadKPIData();
    loadRecentPredictions();
    
    // Refresh data every 30 seconds
    const interval = setInterval(() => {
      checkApiHealth();
      loadPerformanceData();
      loadFeatureStoreStats();
      loadKPIData();
      loadRecentPredictions();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  // Auto-expand category containing active tab
  useEffect(() => {
    const activeCategory = menuCategories.find(cat => 
      cat.items.some(item => item.id === activeTab)
    );
    if (activeCategory) {
      setExpandedCategories(prev => {
        if (!prev[activeCategory.id]) {
          return {
            ...prev,
            [activeCategory.id]: true
          };
        }
        return prev;
      });
    }
  }, [activeTab]);

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

  const loadKPIData = async () => {
    try {
      const data = await creditScoringAPI.getLatestKPIs('daily');
      setKpiData(data);
    } catch (error) {
      console.error('Failed to load KPI data:', error);
    }
  };

  const loadRecentPredictions = async () => {
    try {
      const data = await creditScoringAPI.getPredictions(null, 5, 0);
      setRecentPredictions(data.predictions || []);
    } catch (error) {
      console.error('Failed to load recent predictions:', error);
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
      loadPerformanceData();
      loadRecentPredictions();
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

  const stats = performanceData?.stats?.all || {};
  const p95Latency = stats.p95 || 0;
  const totalRequests = stats.count || 0;
  const errorRate = performanceData?.stats?.error_rate?.rate || 0;
  const avgLatency = stats.avg || 0;

  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const toggleCategory = (categoryId) => {
    setExpandedCategories(prev => ({
      ...prev,
      [categoryId]: !prev[categoryId]
    }));
  };

  // Calculate additional metrics
  const approvalRate = kpiData?.approval_rate || 0;
  const rejectionRate = kpiData?.rejection_rate || 0;
  const totalRevenue = kpiData?.total_revenue || 0;
  const totalPredictions = kpiData?.total_predictions || totalRequests;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Top Navigation Bar */}
      <header className="bg-white/80 backdrop-blur-lg border-b border-slate-200/50 shadow-sm sticky top-0 z-50">
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <Menu className="w-6 h-6 text-slate-600" />
              </button>
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl shadow-lg">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    Credit Risk Scoring
                  </h1>
                  <p className="text-xs text-slate-500">AI-Powered Decision Platform</p>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-2 bg-gradient-to-r from-slate-50 to-blue-50 px-4 py-2 rounded-lg border border-slate-200">
                {getStatusIcon()}
                <span className="text-sm font-medium text-slate-700">
                  {apiStatus === 'healthy' ? 'System Online' : 
                   apiStatus === 'degraded' ? 'Degraded' : 'Offline'}
                </span>
              </div>
              {user && (
                <div className="flex items-center gap-3">
                  <div className="text-right hidden sm:block">
                    <div className="text-sm font-semibold text-slate-800">{user.full_name || user.username}</div>
                    <div className="text-xs text-slate-500">{user.department || 'User'}</div>
                  </div>
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-full flex items-center justify-center shadow-md">
                    <UserIcon className="w-5 h-5 text-white" />
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
        {/* Enhanced Sidebar Navigation */}
        <aside className={`
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          fixed lg:static lg:translate-x-0
          w-72 bg-white/95 backdrop-blur-lg border-r border-slate-200/50 h-[calc(100vh-73px)]
          transition-transform duration-300 z-40
          overflow-y-auto shadow-lg lg:shadow-none
        `}>
          <div className="px-4 py-6">
              <div className="flex items-center justify-between mb-8 lg:hidden">
              <h2 className="text-lg font-bold text-slate-800">Navigation</h2>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <XIcon className="w-5 h-5 text-slate-600" />
              </button>
            </div>
            <nav className="space-y-5">
              {menuCategories.map((category) => {
                const isExpanded = expandedCategories[category.id];
                const hasActiveItem = category.items.some(item => activeTab === item.id);
                
                return (
                  <div key={category.id}>
                    <button
                      onClick={() => toggleCategory(category.id)}
                      className={`
                        w-full flex items-center justify-between px-4 py-2.5 mb-3 rounded-lg
                        transition-all duration-200 group
                        ${hasActiveItem 
                          ? 'bg-blue-50 text-blue-700' 
                          : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                        }
                      `}
                    >
                      <h3 className="text-xs font-bold uppercase tracking-wider">
                        {category.label}
                      </h3>
                      {isExpanded ? (
                        <ChevronUp className={`w-4 h-4 transition-transform ${hasActiveItem ? 'text-blue-600' : 'text-slate-400 group-hover:text-slate-600'}`} />
                      ) : (
                        <ChevronDown className={`w-4 h-4 transition-transform ${hasActiveItem ? 'text-blue-600' : 'text-slate-400 group-hover:text-slate-600'}`} />
                      )}
                    </button>
                    <div className={`
                      space-y-1 overflow-hidden transition-all duration-300 ease-in-out
                      ${isExpanded ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'}
                    `}>
                      {category.items.map((item) => {
                const Icon = item.icon;
                        const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActiveTab(item.id);
                      if (window.innerWidth < 1024) setSidebarOpen(false);
                    }}
                    className={`
                              w-full flex items-center gap-3 px-4 py-2.5 rounded-xl
                              transition-all duration-200 text-left group
                              ${isActive
                                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/30'
                                : 'text-slate-700 hover:bg-slate-100 hover:shadow-md'
                      }
                    `}
                  >
                            <Icon className={`w-5 h-5 ${isActive ? 'text-white' : 'text-slate-500 group-hover:text-blue-600'}`} />
                            <span className="font-medium flex-1">{item.label}</span>
                            {isActive && (
                              <div className="w-2 h-2 bg-white rounded-full"></div>
                    )}
                  </button>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </nav>
          </div>
        </aside>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div
            className="lg:hidden fixed inset-0 bg-black/20 z-30 backdrop-blur-sm"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 px-4 sm:px-6 lg:px-10 xl:px-12 py-6 lg:py-8 max-w-7xl mx-auto w-full">
          {/* Enhanced Overview Section */}
          {activeTab === 'overview' && (
            <div className="space-y-8">
              {/* Hero Stats Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                {/* Total Predictions */}
                <div className="card bg-gradient-to-br from-blue-600 via-blue-500 to-indigo-600 text-white relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16"></div>
                  <div className="relative z-10">
                  <div className="flex items-center justify-between mb-4">
                      <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                      <Target className="w-6 h-6" />
                    </div>
                      <TrendingUp className="w-5 h-5 opacity-80" />
                  </div>
                    <div className="text-4xl font-bold mb-1">{totalPredictions.toLocaleString()}</div>
                    <div className="text-sm opacity-90">Total Predictions</div>
                    <div className="mt-2 text-xs opacity-75">All time processed</div>
                  </div>
                </div>

                {/* P95 Latency */}
                <div className="card bg-gradient-to-br from-emerald-500 via-green-500 to-teal-600 text-white relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16"></div>
                  <div className="relative z-10">
                  <div className="flex items-center justify-between mb-4">
                      <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                        <Zap className="w-6 h-6" />
                    </div>
                      {p95Latency > 0 && p95Latency < 200 ? (
                        <CheckCircle className="w-5 h-5 opacity-80" />
                      ) : (
                        <Clock className="w-5 h-5 opacity-80" />
                      )}
                  </div>
                    <div className="text-4xl font-bold mb-1">
                      {p95Latency > 0 ? `${p95Latency.toFixed(0)}ms` : 'N/A'}
                  </div>
                    <div className="text-sm opacity-90">P95 Latency</div>
                    <div className="mt-2 text-xs opacity-75">
                    {p95Latency > 0 && p95Latency < 200 ? '✓ SLA Compliant' : 'No data yet'}
                    </div>
                  </div>
                </div>

                {/* Approval Rate */}
                <div className="card bg-gradient-to-br from-purple-500 via-pink-500 to-rose-500 text-white relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16"></div>
                  <div className="relative z-10">
                  <div className="flex items-center justify-between mb-4">
                      <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                        <CheckCircle className="w-6 h-6" />
                    </div>
                      <TrendingUp className="w-5 h-5 opacity-80" />
                  </div>
                    <div className="text-4xl font-bold mb-1">
                      {(approvalRate * 100).toFixed(1)}%
                  </div>
                    <div className="text-sm opacity-90">Approval Rate</div>
                    <div className="mt-2 text-xs opacity-75">Auto-approved applications</div>
                  </div>
                </div>

                {/* System Status */}
                <div className="card bg-gradient-to-br from-amber-500 via-orange-500 to-red-500 text-white relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16"></div>
                  <div className="relative z-10">
                  <div className="flex items-center justify-between mb-4">
                      <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                        <Server className="w-6 h-6" />
                    </div>
                      {getStatusIcon()}
                  </div>
                    <div className="text-4xl font-bold mb-1">
                      {apiStatus === 'healthy' ? 'Active' : apiStatus === 'degraded' ? 'Degraded' : 'Offline'}
                  </div>
                    <div className="text-sm opacity-90">System Status</div>
                    <div className="mt-2 text-xs opacity-75">Production ready</div>
                  </div>
                </div>
                </div>

              {/* Secondary Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="card bg-white border-2 border-slate-200 hover:border-blue-300 transition-colors">
                  <div className="flex items-center justify-between mb-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <DollarSign className="w-5 h-5 text-blue-600" />
                    </div>
                    <span className="text-xs font-semibold text-slate-500">Revenue Impact</span>
                  </div>
                  <div className="text-2xl font-bold text-slate-800 mb-1">
                    ${(totalRevenue / 1000).toFixed(1)}K
                  </div>
                  <div className="text-xs text-slate-600">From approved applications</div>
                </div>

                <div className="card bg-white border-2 border-slate-200 hover:border-red-300 transition-colors">
                  <div className="flex items-center justify-between mb-3">
                    <div className="p-2 bg-red-100 rounded-lg">
                      <XCircle className="w-5 h-5 text-red-600" />
                    </div>
                    <span className="text-xs font-semibold text-slate-500">Rejection Rate</span>
                  </div>
                  <div className="text-2xl font-bold text-slate-800 mb-1">
                    {(rejectionRate * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-600">Auto-rejected applications</div>
                </div>

                <div className="card bg-white border-2 border-slate-200 hover:border-green-300 transition-colors">
                  <div className="flex items-center justify-between mb-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Activity className="w-5 h-5 text-green-600" />
                    </div>
                    <span className="text-xs font-semibold text-slate-500">Avg Latency</span>
                  </div>
                  <div className="text-2xl font-bold text-slate-800 mb-1">
                    {avgLatency > 0 ? `${avgLatency.toFixed(0)}ms` : 'N/A'}
                  </div>
                  <div className="text-xs text-slate-600">Average response time</div>
                </div>

                <div className="card bg-white border-2 border-slate-200 hover:border-purple-300 transition-colors">
                  <div className="flex items-center justify-between mb-3">
                    <div className="p-2 bg-purple-100 rounded-lg">
                      <Database className="w-5 h-5 text-purple-600" />
                    </div>
                    <span className="text-xs font-semibold text-slate-500">Feature Store</span>
                  </div>
                  <div className="text-2xl font-bold text-slate-800 mb-1">
                    {featureStoreStats?.total_features?.toLocaleString() || 0}
                  </div>
                  <div className="text-xs text-slate-600">
                    {featureStoreStats?.cache_coverage 
                      ? `${featureStoreStats.cache_coverage.toFixed(0)}% cached`
                      : 'Cached features'}
                  </div>
                </div>
              </div>

              {/* Welcome Section with Quick Actions */}
              <div className="card bg-gradient-to-br from-white via-blue-50 to-indigo-50 border-2 border-blue-200/50 shadow-xl">
                <div className="flex flex-col lg:flex-row items-center justify-between gap-6">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-2 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg">
                        <Sparkles className="w-6 h-6 text-white" />
                      </div>
                      <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                        Welcome to Credit Risk Platform
                    </h2>
                    </div>
                    <p className="text-slate-700 text-lg mb-4 leading-relaxed">
                      AI-powered credit risk assessment for real-time lending decisions. 
                      Built with industry-leading ML models and regulatory compliance.
                    </p>
                    <div className="flex flex-wrap gap-3">
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-green-100 rounded-full text-sm text-green-700 font-medium">
                        <CheckCircle className="w-4 h-4" />
                        <span>Basel II Compliant</span>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-100 rounded-full text-sm text-blue-700 font-medium">
                        <Zap className="w-4 h-4" />
                        <span>Sub-200ms Latency</span>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-100 rounded-full text-sm text-purple-700 font-medium">
                        <Brain className="w-4 h-4" />
                        <span>SHAP Explanations</span>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-pink-100 rounded-full text-sm text-pink-700 font-medium">
                        <Shield className="w-4 h-4" />
                        <span>Fairness Monitoring</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col gap-3">
                  <button
                    onClick={() => setActiveTab('predict')}
                      className="btn-primary flex items-center justify-center gap-2 px-8 py-4 text-lg shadow-xl hover:shadow-2xl"
                  >
                      <Brain className="w-5 h-5" />
                    Start Assessment
                    <ArrowRight className="w-5 h-5" />
                  </button>
                    <button
                      onClick={() => setActiveTab('kpis')}
                      className="btn-secondary flex items-center justify-center gap-2 px-8 py-4 text-lg"
                    >
                      <BarChart3 className="w-5 h-5" />
                      View Analytics
                    </button>
                  </div>
                </div>
              </div>

              {/* Quick Actions Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div 
                  className="card hover:shadow-2xl transition-all cursor-pointer group border-2 border-transparent hover:border-blue-300"
                  onClick={() => setActiveTab('predict')}
                >
                  <div className="flex items-center gap-4">
                    <div className="p-4 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-xl group-hover:scale-110 transition-transform">
                      <Brain className="w-8 h-8 text-white" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-800 group-hover:text-blue-600 transition-colors">Risk Assessment</h3>
                      <p className="text-sm text-slate-600">Evaluate customer credit risk</p>
                    </div>
                  </div>
                </div>

                <div 
                  className="card hover:shadow-2xl transition-all cursor-pointer group border-2 border-transparent hover:border-green-300"
                  onClick={() => setActiveTab('performance')}
                >
                  <div className="flex items-center gap-4">
                    <div className="p-4 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl group-hover:scale-110 transition-transform">
                      <Activity className="w-8 h-8 text-white" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-800 group-hover:text-green-600 transition-colors">Performance</h3>
                      <p className="text-sm text-slate-600">Monitor system metrics</p>
                    </div>
                  </div>
                </div>

                <div 
                  className="card hover:shadow-2xl transition-all cursor-pointer group border-2 border-transparent hover:border-purple-300"
                  onClick={() => setActiveTab('governance')}
                >
                  <div className="flex items-center gap-4">
                    <div className="p-4 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl group-hover:scale-110 transition-transform">
                      <Shield className="w-8 h-8 text-white" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-800 group-hover:text-purple-600 transition-colors">Governance</h3>
                      <p className="text-sm text-slate-600">Model compliance & fairness</p>
                    </div>
                  </div>
                </div>

                <div 
                  className="card hover:shadow-2xl transition-all cursor-pointer group border-2 border-transparent hover:border-amber-300"
                  onClick={() => setActiveTab('kpis')}
                >
                  <div className="flex items-center gap-4">
                    <div className="p-4 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl group-hover:scale-110 transition-transform">
                      <BarChart3 className="w-8 h-8 text-white" />
              </div>
                    <div>
                      <h3 className="font-bold text-slate-800 group-hover:text-amber-600 transition-colors">Business KPIs</h3>
                      <p className="text-sm text-slate-600">Revenue & analytics</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              {recentPredictions.length > 0 && (
                <div className="card">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                      <Clock className="w-5 h-5 text-blue-600" />
                      Recent Activity
                    </h3>
                    <button
                      onClick={() => setActiveTab('predictions')}
                      className="text-sm text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1"
                    >
                      View All
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="space-y-3">
                    {recentPredictions.slice(0, 5).map((pred, idx) => (
                      <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors">
                        <div className="flex items-center gap-3">
                          <div className={`w-2 h-2 rounded-full ${
                            pred.risk_level === 'low' ? 'bg-green-500' :
                            pred.risk_level === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                          }`}></div>
                          <div>
                            <div className="font-medium text-slate-800">
                              {pred.customer_id || 'Unknown Customer'}
                            </div>
                            <div className="text-xs text-slate-500">
                              {pred.predicted_at ? new Date(pred.predicted_at).toLocaleString() : 'Just now'}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`font-bold ${
                            pred.risk_level === 'low' ? 'text-green-600' :
                            pred.risk_level === 'medium' ? 'text-yellow-600' : 'text-red-600'
                          }`}>
                            {(pred.risk_score * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-slate-500 capitalize">{pred.risk_level || 'Unknown'}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* All other sections remain the same but with enhanced headers */}
          {activeTab === 'data' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Data Upload</h2>
                  <p className="text-slate-600 mt-1">Upload raw transaction data to the database</p>
                </div>
              </div>
              <DataUpload />
            </div>
          )}

          {activeTab === 'transactions' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Transaction Data</h2>
                  <p className="text-slate-600 mt-1">View and filter uploaded transaction data from the database</p>
                </div>
              </div>
              <TransactionsTable />
            </div>
          )}

          {activeTab === 'predict' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Risk Assessment</h2>
                  <p className="text-slate-600 mt-1">Evaluate customer credit risk in real-time</p>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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

                <div className="lg:col-span-2 space-y-6">
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

                  {resultTab === 'result' && prediction && (
                    <PredictionResult prediction={prediction} />
                  )}

                  {resultTab === 'explanation' && explanation && (
                    <ExplanationPanel explanation={explanation} />
                  )}

                  {resultTab === 'scenario' && prediction && explanation && (
                    <ScenarioTester
                      initialFeatures={features}
                      featureNames={featureNames}
                      currentPrediction={prediction}
                    />
                  )}

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

          {/* All other sections with enhanced headers */}
          {activeTab === 'performance' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Performance Monitor</h2>
                  <p className="text-slate-600 mt-1">Real-time system metrics and SLA compliance</p>
                </div>
              </div>
              <PerformanceMonitor />
            </div>
          )}

          {activeTab === 'model-validation' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Model Performance & Validation</h2>
                  <p className="text-slate-600 mt-1">Statistical rigor, proper evaluation, and production readiness</p>
                </div>
              </div>
              <ModelPerformanceValidation />
            </div>
          )}

          {activeTab === 'governance' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Model Governance</h2>
                  <p className="text-slate-600 mt-1">Compliance, fairness, and regulatory oversight</p>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-6">
                <ModelCard />
                <FairnessAnalysis />
              </div>
            </div>
          )}

          {activeTab === 'predictions' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Predictions History</h2>
                  <p className="text-slate-600 mt-1">View all predictions stored in the database</p>
                </div>
              </div>
              <PredictionsTable />
            </div>
          )}

          {activeTab === 'scores' && (
            <div className="space-y-8">
              <CustomerScoresTable />
            </div>
          )}

          {activeTab === 'score-customer' && (
            <div className="space-y-8">
              <CustomerScorer />
            </div>
          )}

          {activeTab === 'kpis' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Business KPIs</h2>
                  <p className="text-slate-600 mt-1">Key performance indicators and analytics</p>
                </div>
              </div>
              <BusinessKPIs />
            </div>
          )}

          {activeTab === 'users' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Users Management</h2>
                  <p className="text-slate-600 mt-1">Manage system users and their access</p>
                </div>
              </div>
              <UsersTable />
            </div>
          )}

          {activeTab === 'roles' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Roles & Permissions</h2>
                  <p className="text-slate-600 mt-1">Manage roles and their permissions</p>
                </div>
              </div>
              <RolesTable />
            </div>
          )}

          {activeTab === 'versions' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Version Information</h2>
                  <p className="text-slate-600 mt-1">Model and data version tracking</p>
                </div>
              </div>
              <VersionInfo />
            </div>
          )}

          {activeTab === 'drift' && (
            <div className="space-y-8">
              <DriftDetection />
            </div>
          )}

          {activeTab === 'alerts' && (
            <div className="space-y-8">
              <AlertsPanel />
            </div>
          )}

          {activeTab === 'lineage' && (
            <div className="space-y-8">
              <DataLineage />
            </div>
          )}

          {activeTab === 'data-quality' && (
            <div className="space-y-8">
              <DataQualityMonitor />
            </div>
          )}

          {activeTab === 'feature-store' && (
            <div className="space-y-8">
              <FeatureStore />
            </div>
          )}

          {activeTab === 'ab-testing' && (
            <div className="space-y-8">
              <ABTesting />
            </div>
          )}

          {activeTab === 'retraining' && (
            <div className="space-y-8">
              <ModelRetraining />
            </div>
          )}

          {activeTab === 'batch-predictions' && (
            <div className="space-y-8">
              <BatchPredictions />
            </div>
          )}

          {activeTab === 'load-testing' && (
            <div className="space-y-8">
              <LoadTesting />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
