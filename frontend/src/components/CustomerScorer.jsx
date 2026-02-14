import React, { useState, useEffect } from 'react';
import { Search, User, TrendingUp, Award, RefreshCw, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const CustomerScorer = () => {
  const [customers, setCustomers] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

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
          transactions: transactions
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to score customer');
      }
      
      const scoreResult = await response.json();
      setResult(scoreResult);
      
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
        <div className="p-6 bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle className="w-6 h-6 text-green-600" />
            <h3 className="text-lg font-semibold text-gray-900">Score Generated Successfully</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Customer ID</p>
              <p className="text-lg font-semibold text-gray-900">{result.customer_id}</p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Credit Score</p>
              <p className={`text-3xl font-bold ${getScoreColor(result.customer_score)}`}>
                {result.customer_score}
                <span className="text-lg text-gray-500">/100</span>
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Risk Level</p>
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getRiskBadgeClass(result.risk_level)}`}>
                {result.risk_level?.toUpperCase() || 'UNKNOWN'}
              </span>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Probability:</span>
              <span className="ml-2 font-semibold text-gray-900">
                {(result.probability * 100).toFixed(2)}%
              </span>
            </div>
            <div>
              <span className="text-gray-600">Prediction ID:</span>
              <span className="ml-2 font-mono text-xs text-gray-700">
                {result.prediction_id}
              </span>
            </div>
          </div>
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
          <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
            {filteredCustomers.map((customer) => (
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
        )}
      </div>
    </div>
  );
};

export default CustomerScorer;
