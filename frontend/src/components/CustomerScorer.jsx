import React, { useState, useEffect } from 'react';
import { Search, User, TrendingUp, Award, RefreshCw, CheckCircle, AlertCircle, Loader, ChevronLeft, ChevronRight, Brain, FileText, Shield, Eye } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';
import ExplanationPanel from './ExplanationPanel';

const CustomerScorer = () => {
  const [customers, setCustomers] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);
  const [features, setFeatures] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 5;

  useEffect(() => {
    loadCustomers();
  }, []);

  const loadCustomers = async () => {
    setLoading(true);
    setError(null);
    try {
      // Get transactions to extract unique customers
      const data = await creditScoringAPI.getTransactions(null, 1000, 0);
      const transactions = data.transactions || [];
      
      // Extract unique customer IDs
      const uniqueCustomers = [...new Set(transactions.map(t => t.customer_id).filter(Boolean))];
      
      // Get customer transaction counts
      const customerCounts = {};
      transactions.forEach(t => {
        if (t.customer_id) {
          customerCounts[t.customer_id] = (customerCounts[t.customer_id] || 0) + 1;
        }
      });
      
      // Sort by transaction count (most active first)
      const sortedCustomers = uniqueCustomers
        .map(customerId => ({
          customer_id: customerId,
          transaction_count: customerCounts[customerId] || 0
        }))
        .sort((a, b) => b.transaction_count - a.transaction_count);
      
      setCustomers(sortedCustomers);
    } catch (err) {
      setError(err.message || 'Failed to load customers');
      console.error('Error loading customers:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleScoreCustomer = async (customerId) => {
    setScoring(true);
    setError(null);
    setResult(null);
    setSelectedCustomer(customerId);
    
    try {
      // First, get the customer's transactions
      const txnData = await creditScoringAPI.getTransactions(customerId, 1000, 0);
      const transactions = txnData.transactions || [];
      
      if (transactions.length === 0) {
        throw new Error('No transactions found for this customer');
      }
      
      // Call the backend API to generate features and score
      const response = await fetch('/api/customers/score', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({
          customer_id: customerId,
          transactions: transactions,
          include_features: true,  // Request features for explanation
          include_explanation: true  // Request explanation automatically
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to score customer');
      }
      
      const scoreResult = await response.json();
      setResult(scoreResult);
      
      // Store features if available in response
      if (scoreResult.features) {
        setFeatures(scoreResult.features);
        console.log('Features received:', scoreResult.features.length, 'features');
      } else {
        console.warn('No features in response');
      }
      
      // If explanation is included in response, use it and show it
      if (scoreResult.explanation) {
        console.log('Explanation received:', scoreResult.explanation);
        setExplanation(scoreResult.explanation);
        setShowExplanation(true);  // Automatically show explanation if available
      } else {
        console.log('No explanation in response, will generate on demand');
        setExplanation(null);
        setShowExplanation(false);
      }
      
      // Refresh customer scores table after scoring
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('customerScored', { detail: scoreResult }));
      }, 1000);
      
    } catch (err) {
      setError(err.message || 'Failed to score customer');
      console.error('Error scoring customer:', err);
    } finally {
      setScoring(false);
    }
  };

  const filteredCustomers = customers.filter(customer =>
    customer.customer_id?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Pagination calculations
  const totalPages = Math.ceil(filteredCustomers.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedCustomers = filteredCustomers.slice(startIndex, endIndex);

  // Reset to page 1 when search term changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm]);

  const handlePreviousPage = () => {
    setCurrentPage(prev => Math.max(1, prev - 1));
  };

  const handleNextPage = () => {
    setCurrentPage(prev => Math.min(totalPages, prev + 1));
  };

  const handlePageClick = (page) => {
    setCurrentPage(page);
  };

  const getScoreColor = (score) => {
    if (score >= 70) return 'text-green-600';
    if (score >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBg = (score) => {
    if (score >= 70) return 'bg-green-100 text-green-800 border-green-200';
    if (score >= 40) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  const getRiskBadgeClass = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getDecision = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return { text: 'APPROVE', color: 'green', icon: CheckCircle };
      case 'medium':
        return { text: 'REVIEW', color: 'yellow', icon: AlertCircle };
      case 'high':
        return { text: 'REJECT', color: 'red', icon: AlertCircle };
      default:
        return { text: 'UNKNOWN', color: 'gray', icon: AlertCircle };
    }
  };

  const handleGetExplanation = async () => {
    if (!result || !features) {
      setError('Features not available for explanation. Please score the customer again.');
      console.error('Cannot generate explanation: missing result or features', { hasResult: !!result, hasFeatures: !!features });
      return;
    }

    setLoadingExplanation(true);
    setError(null);
    try {
      console.log('Generating explanation with features:', features.length, 'features');
      const explanationData = await creditScoringAPI.explain(features);
      console.log('Explanation generated:', explanationData);
      setExplanation(explanationData);
      setShowExplanation(true);
    } catch (err) {
      const errorMsg = 'Failed to generate explanation: ' + (err.message || 'Unknown error');
      setError(errorMsg);
      console.error('Error getting explanation:', err);
    } finally {
      setLoadingExplanation(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Score a Customer</h2>
          <p className="text-sm text-gray-600 mt-1">Select a customer and generate their credit score</p>
        </div>
        <button
          onClick={loadCustomers}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-400">
          <div className="flex items-center gap-2 text-red-700">
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="space-y-6">
          {/* Main Result Card */}
          <div className="p-6 bg-white rounded-lg shadow-sm border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-6 h-6 text-green-600" />
                <h3 className="text-lg font-semibold text-gray-900">Score Generated Successfully</h3>
              </div>
              <button
                onClick={async () => {
                  if (!showExplanation) {
                    // If showing explanation and it doesn't exist, generate it
                    if (!explanation && features) {
                      await handleGetExplanation();
                    } else {
                      setShowExplanation(true);
                    }
                  } else {
                    setShowExplanation(false);
                  }
                }}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-indigo-700 bg-indigo-50 border border-indigo-200 rounded-md hover:bg-indigo-100 disabled:opacity-50"
                disabled={loadingExplanation}
              >
                {loadingExplanation ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4" />
                    {showExplanation ? 'Hide' : 'Show'} Explanation
                  </>
                )}
              </button>
            </div>
          
          {/* Data Sufficiency Warning */}
          {result.data_sufficiency_warning && (
            <div className="mb-4 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-yellow-800">Limited Transaction History</p>
                  <p className="text-sm text-yellow-700 mt-1">{result.data_sufficiency_warning}</p>
                </div>
              </div>
            </div>
          )}

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
              <p className="text-sm text-gray-600 mb-1 flex items-center gap-1">
                <User className="w-4 h-4" />
                Customer ID
              </p>
              <p className="text-lg font-semibold text-gray-900 font-mono">{result.customer_id}</p>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border border-purple-200">
              <p className="text-sm text-gray-600 mb-1 flex items-center gap-1">
                <Award className="w-4 h-4" />
                Credit Score
              </p>
              <p className={`text-3xl font-bold ${getScoreColor(result.customer_score)}`}>
                {result.customer_score}
                <span className="text-lg text-gray-500">/100</span>
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-lg border border-indigo-200">
              <p className="text-sm text-gray-600 mb-1 flex items-center gap-1">
                <Shield className="w-4 h-4" />
                Risk Level
              </p>
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getRiskBadgeClass(result.risk_level)}`}>
                {result.risk_level?.toUpperCase() || 'UNKNOWN'}
              </span>
            </div>
            <div className="p-4 bg-gradient-to-br from-amber-50 to-amber-100 rounded-lg border border-amber-200">
              <p className="text-sm text-gray-600 mb-1 flex items-center gap-1">
                <FileText className="w-4 h-4" />
                Features Used
              </p>
              <p className="text-2xl font-bold text-gray-900">
                {result.features_used || 'N/A'}
              </p>
            </div>
          </div>

          {/* Decision Section */}
          {(() => {
            const decision = getDecision(result.risk_level);
            const DecisionIcon = decision.icon;
            return (
              <div className={`mb-6 p-6 rounded-lg border-2 ${
                decision.color === 'green' ? 'bg-green-50 border-green-300' :
                decision.color === 'yellow' ? 'bg-yellow-50 border-yellow-300' :
                'bg-red-50 border-red-300'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-full ${
                      decision.color === 'green' ? 'bg-green-200' :
                      decision.color === 'yellow' ? 'bg-yellow-200' :
                      'bg-red-200'
                    }`}>
                      <DecisionIcon className={`w-8 h-8 ${
                        decision.color === 'green' ? 'text-green-700' :
                        decision.color === 'yellow' ? 'text-yellow-700' :
                        'text-red-700'
                      }`} />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-600 mb-1">Decision</p>
                      <p className={`text-3xl font-bold ${
                        decision.color === 'green' ? 'text-green-700' :
                        decision.color === 'yellow' ? 'text-yellow-700' :
                        'text-red-700'
                      }`}>
                        {decision.text}
                      </p>
                      <p className="text-sm text-gray-600 mt-1">
                        {decision.text === 'APPROVE' && 'Customer meets low-risk criteria. Auto-approve recommended.'}
                        {decision.text === 'REVIEW' && 'Customer requires manual review. Additional verification may be needed.'}
                        {decision.text === 'REJECT' && 'Customer shows high-risk indicators. Auto-reject recommended.'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Probability of High Risk:</span>
              <span className="ml-2 font-semibold text-gray-900">
                {(result.probability * 100).toFixed(2)}%
              </span>
            </div>
            <div>
              <span className="text-gray-600">Transactions Used:</span>
              <span className="ml-2 font-semibold text-gray-900">
                {result.transaction_count || 'N/A'}
              </span>
            </div>
            {result.prediction_quality && (
              <>
                <div>
                  <span className="text-gray-600">Confidence Score:</span>
                  <span className={`ml-2 font-semibold ${
                    result.prediction_quality.confidence_score >= 0.8 ? 'text-green-600' :
                    result.prediction_quality.confidence_score >= 0.5 ? 'text-yellow-600' :
                    'text-red-600'
                  }`}>
                    {(result.prediction_quality.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Uncertainty Level:</span>
                  <span className={`ml-2 font-semibold ${
                    result.prediction_quality.uncertainty_level === 'low' ? 'text-green-600' :
                    result.prediction_quality.uncertainty_level === 'medium' ? 'text-yellow-600' :
                    'text-red-600'
                  }`}>
                    {result.prediction_quality.uncertainty_level.toUpperCase()}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Data Quality:</span>
                  <span className={`ml-2 font-semibold ${
                    result.prediction_quality.data_quality_score >= 0.8 ? 'text-green-600' :
                    result.prediction_quality.data_quality_score >= 0.5 ? 'text-yellow-600' :
                    'text-red-600'
                  }`}>
                    {(result.prediction_quality.data_quality_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Feature Completeness:</span>
                  <span className="ml-2 font-semibold text-gray-900">
                    {(result.prediction_quality.feature_completeness * 100).toFixed(1)}%
                  </span>
                </div>
              </>
            )}
            <div className="col-span-2">
              <span className="text-gray-600">Prediction ID:</span>
              <span className="ml-2 font-mono text-xs text-gray-700">
                {result.prediction_id}
              </span>
            </div>
          </div>

          {/* Customer Input Features Section */}
          <div className="mt-6 p-4 bg-slate-50 rounded-lg border border-slate-200">
            <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <Eye className="w-4 h-4" />
              Customer Input Features
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <span className="text-gray-600">Transaction Count:</span>
                <span className="ml-2 font-semibold text-gray-900">{result.transaction_count || 'N/A'}</span>
              </div>
              <div>
                <span className="text-gray-600">Features Used:</span>
                <span className="ml-2 font-semibold text-gray-900">{result.features_used || 'N/A'}</span>
              </div>
              <div>
                <span className="text-gray-600">From Feature Store:</span>
                <span className="ml-2 font-semibold text-gray-900">
                  {result.features_from_store ? 'Yes' : 'No'}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Timestamp:</span>
                <span className="ml-2 font-semibold text-gray-900 text-xs">
                  {result.timestamp ? new Date(result.timestamp).toLocaleString() : 'N/A'}
                </span>
              </div>
            </div>
            {features && (
              <div className="mt-3 pt-3 border-t border-slate-300">
                <p className="text-xs text-gray-600 mb-2">Feature values available for explanation</p>
                <button
                  onClick={handleGetExplanation}
                  disabled={loadingExplanation}
                  className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 disabled:opacity-50"
                >
                  {loadingExplanation ? (
                    <>
                      <Loader className="w-4 h-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4" />
                      Generate SHAP Explanation
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Explanation Panel */}
        {showExplanation && explanation && (
          <div className="mt-6">
            <ExplanationPanel explanation={explanation} />
          </div>
        )}

        {showExplanation && !explanation && !loadingExplanation && (
          <div className="mt-6 p-6 bg-indigo-50 rounded-lg border border-indigo-200">
            {features ? (
              <div className="text-center">
                <Brain className="w-12 h-12 text-indigo-600 mx-auto mb-3" />
                <p className="text-gray-700 mb-4">Explanation not yet generated</p>
                <button
                  onClick={handleGetExplanation}
                  disabled={loadingExplanation}
                  className="flex items-center gap-2 px-6 py-3 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 disabled:opacity-50 mx-auto"
                >
                  {loadingExplanation ? (
                    <>
                      <Loader className="w-4 h-4 animate-spin" />
                      Generating Explanation...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4" />
                      Generate SHAP Explanation
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="text-center">
                <AlertCircle className="w-12 h-12 text-yellow-600 mx-auto mb-3" />
                <p className="text-gray-700 mb-2">Features not available</p>
                <p className="text-sm text-gray-600">Please score the customer again to generate explanation</p>
              </div>
            )}
          </div>
        )}

        {showExplanation && loadingExplanation && (
          <div className="mt-6 p-6 bg-indigo-50 rounded-lg border border-indigo-200 text-center">
            <Loader className="w-12 h-12 text-indigo-600 mx-auto mb-3 animate-spin" />
            <p className="text-gray-700">Generating explanation...</p>
          </div>
        )}
      </div>
      )}

      {/* Customer Selection */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search customers by ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
              />
            </div>
            <div className="text-sm text-gray-600">
              {filteredCustomers.length} customer{filteredCustomers.length !== 1 ? 's' : ''}
              {totalPages > 1 && (
                <span className="ml-2 text-gray-500">
                  (Page {currentPage} of {totalPages})
                </span>
              )}
            </div>
          </div>
        </div>

        {loading ? (
          <div className="p-8 text-center text-gray-500">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
            <p>Loading customers...</p>
          </div>
        ) : filteredCustomers.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <User className="w-12 h-12 mx-auto mb-2 text-gray-400" />
            <p>No customers found</p>
          </div>
        ) : (
          <>
            <div className="divide-y divide-gray-200">
              {paginatedCustomers.map((customer) => (
              <div
                key={customer.customer_id}
                className="p-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <User className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{customer.customer_id}</p>
                      <p className="text-sm text-gray-500">
                        {customer.transaction_count} transaction{customer.transaction_count !== 1 ? 's' : ''}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleScoreCustomer(customer.customer_id)}
                    disabled={scoring && selectedCustomer === customer.customer_id}
                    className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      scoring && selectedCustomer === customer.customer_id
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-blue-600 text-white hover:bg-blue-700'
                    }`}
                  >
                    {scoring && selectedCustomer === customer.customer_id ? (
                      <>
                        <Loader className="w-4 h-4 animate-spin" />
                        Scoring...
                      </>
                    ) : (
                      <>
                        <Award className="w-4 h-4" />
                        Score Customer
                      </>
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="px-4 py-4 border-t border-gray-200 bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-600">
                    Showing {startIndex + 1} to {Math.min(endIndex, filteredCustomers.length)} of {filteredCustomers.length} customers
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handlePreviousPage}
                      disabled={currentPage === 1}
                      className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                        currentPage === 1
                          ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                          : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <ChevronLeft className="w-4 h-4" />
                      Previous
                    </button>
                    
                    {/* Page Numbers */}
                    <div className="flex items-center gap-1">
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        let pageNum;
                        if (totalPages <= 5) {
                          pageNum = i + 1;
                        } else if (currentPage <= 3) {
                          pageNum = i + 1;
                        } else if (currentPage >= totalPages - 2) {
                          pageNum = totalPages - 4 + i;
                        } else {
                          pageNum = currentPage - 2 + i;
                        }
                        
                        return (
                          <button
                            key={pageNum}
                            onClick={() => handlePageClick(pageNum)}
                            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                              currentPage === pageNum
                                ? 'bg-blue-600 text-white'
                                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                            }`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                    </div>

                    <button
                      onClick={handleNextPage}
                      disabled={currentPage === totalPages}
                      className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                        currentPage === totalPages
                          ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                          : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      Next
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default CustomerScorer;
