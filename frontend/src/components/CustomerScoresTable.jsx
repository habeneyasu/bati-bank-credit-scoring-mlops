import React, { useState, useEffect } from 'react';
import { Search, RefreshCw, TrendingUp, TrendingDown, User, Award, Filter, Download } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const CustomerScoresTable = () => {
  const [scores, setScores] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('score'); // 'score', 'customer_id', 'date'
  const [sortOrder, setSortOrder] = useState('desc'); // 'asc', 'desc'
  const [limit, setLimit] = useState(1000); // Increased limit to get more predictions
  const [offset, setOffset] = useState(0);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [scoreFilter, setScoreFilter] = useState(''); // 'high', 'medium', 'low'
  const [allPredictionsLoaded, setAllPredictionsLoaded] = useState(false);

  useEffect(() => {
    loadScores();
  }, [sortBy, sortOrder]);

  const loadScores = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch predictions in batches to get all unique customers
      let allPredictions = [];
      let currentOffset = 0;
      let batchLimit = 1000;
      let hasMoreData = true;
      
      // Fetch multiple batches to get all predictions
      while (hasMoreData && allPredictions.length < 10000) { // Cap at 10k predictions
        const data = await creditScoringAPI.getPredictions(null, batchLimit, currentOffset);
        const batch = data.predictions || [];
        allPredictions = allPredictions.concat(batch);
        
        hasMoreData = data.has_more || false;
        currentOffset += batchLimit;
        
        // If we got fewer than the limit, we've reached the end
        if (batch.length < batchLimit) {
          hasMoreData = false;
        }
      }
      
      setTotal(allPredictions.length);
      setAllPredictionsLoaded(true);
      
      // Group by customer_id and get the latest score for each customer
      const customerScoresMap = new Map();
      allPredictions.forEach(pred => {
        if (pred.customer_id && pred.customer_score !== null && pred.customer_score !== undefined) {
          const existing = customerScoresMap.get(pred.customer_id);
          if (!existing || new Date(pred.created_at) > new Date(existing.created_at)) {
            customerScoresMap.set(pred.customer_id, {
              customer_id: pred.customer_id,
              customer_score: pred.customer_score,
              risk_level: pred.risk_level,
              probability: pred.probability,
              prediction_id: pred.prediction_id,
              created_at: pred.created_at,
              model_version: pred.model_version
            });
          }
        }
      });
      
      let scoresList = Array.from(customerScoresMap.values());
      
      // Apply filters
      if (searchTerm) {
        const search = searchTerm.toLowerCase();
        scoresList = scoresList.filter(s => 
          s.customer_id?.toLowerCase().includes(search)
        );
      }
      
      if (scoreFilter) {
        scoresList = scoresList.filter(s => {
          if (scoreFilter === 'high') return s.customer_score >= 70;
          if (scoreFilter === 'medium') return s.customer_score >= 40 && s.customer_score < 70;
          if (scoreFilter === 'low') return s.customer_score < 40;
          return true;
        });
      }
      
      // Sort
      scoresList.sort((a, b) => {
        let aVal, bVal;
        switch (sortBy) {
          case 'score':
            aVal = a.customer_score || 0;
            bVal = b.customer_score || 0;
            break;
          case 'customer_id':
            aVal = a.customer_id || '';
            bVal = b.customer_id || '';
            break;
          case 'date':
            aVal = new Date(a.created_at || 0);
            bVal = new Date(b.created_at || 0);
            break;
          default:
            aVal = a.customer_score || 0;
            bVal = b.customer_score || 0;
        }
        
        if (sortOrder === 'asc') {
          return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        } else {
          return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
        }
      });
      
      setScores(scoresList);
      setHasMore(false); // We've loaded all data, no pagination needed
      
      // Log for debugging
      console.log('Customer Scores loaded:', {
        totalPredictions: allPredictions.length,
        uniqueCustomers: scoresList.length,
        withScores: allPredictions.filter(p => p.customer_id && p.customer_score !== null && p.customer_score !== undefined).length
      });
      
    } catch (err) {
      setError(err.message || 'Failed to load customer scores');
      console.error('Error loading customer scores:', err);
    } finally {
      setLoading(false);
    }
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

  const getScoreLabel = (score) => {
    if (score >= 70) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 50) return 'Fair';
    if (score >= 40) return 'Poor';
    return 'Very Poor';
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

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('desc');
    }
  };

  const SortIcon = ({ column }) => {
    if (sortBy !== column) return null;
    return sortOrder === 'asc' ? (
      <TrendingUp className="w-4 h-4 ml-1" />
    ) : (
      <TrendingDown className="w-4 h-4 ml-1" />
    );
  };

  // Calculate statistics
  const stats = scores.length > 0 ? {
    average: scores.reduce((sum, s) => sum + (s.customer_score || 0), 0) / scores.length,
    min: Math.min(...scores.map(s => s.customer_score || 0)),
    max: Math.max(...scores.map(s => s.customer_score || 0)),
    excellent: scores.filter(s => (s.customer_score || 0) >= 70).length,
    good: scores.filter(s => (s.customer_score || 0) >= 60 && (s.customer_score || 0) < 70).length,
    fair: scores.filter(s => (s.customer_score || 0) >= 50 && (s.customer_score || 0) < 60).length,
    poor: scores.filter(s => (s.customer_score || 0) >= 40 && (s.customer_score || 0) < 50).length,
    veryPoor: scores.filter(s => (s.customer_score || 0) < 40).length,
  } : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Customer Credit Scores</h2>
          <p className="text-sm text-gray-600 mt-1">View and analyze customer credit scores (0-100 scale)</p>
        </div>
        <button
          onClick={loadScores}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Average Score</p>
                <p className="text-2xl font-bold text-gray-900">{stats.average.toFixed(0)}</p>
              </div>
              <Award className="w-8 h-8 text-blue-500" />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Score Range</p>
                <p className="text-2xl font-bold text-gray-900">
                  {stats.min} - {stats.max}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-500" />
            </div>
          </div>
            <div className="bg-white rounded-lg shadow-sm border border-green-200 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Excellent (70+)</p>
                  <p className="text-2xl font-bold text-green-600">{stats.excellent}</p>
                </div>
                <Award className="w-8 h-8 text-green-500" />
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-red-200 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Poor (&lt;40)</p>
                  <p className="text-2xl font-bold text-red-600">{stats.veryPoor}</p>
                </div>
                <TrendingDown className="w-8 h-8 text-red-500" />
              </div>
            </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search by Customer ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
            />
          </div>
          <select
            value={scoreFilter}
            onChange={(e) => setScoreFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          >
            <option value="">All Scores</option>
            <option value="high">High (70+)</option>
            <option value="medium">Medium (40-69)</option>
            <option value="low">Low (&lt;40)</option>
          </select>
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <span>Total Customers:</span>
            <span className="font-semibold">{scores.length}</span>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-400">
          <div className="flex items-center gap-2 text-red-700">
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}

      {/* Scores Table */}
      {loading && scores.length === 0 ? (
        <div className="p-8 text-center text-gray-500">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
          <p>Loading customer scores...</p>
        </div>
      ) : scores.length === 0 ? (
        <div className="p-8 text-center text-gray-500 bg-white rounded-lg border border-gray-200">
          <User className="w-12 h-12 mx-auto mb-2 text-gray-400" />
          <p className="font-medium text-gray-900 mb-2">No customer scores found</p>
          <p className="text-sm text-gray-600 mb-4">
            {allPredictionsLoaded 
              ? "No predictions with customer scores are available. Make some predictions with customer_id to see scores here."
              : "Loading predictions..."}
          </p>
          {!allPredictionsLoaded && (
            <button
              onClick={loadScores}
              className="mt-4 px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded-md hover:bg-blue-100"
            >
              Retry Loading
            </button>
          )}
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('customer_id')}
                  >
                    <div className="flex items-center">
                      Customer ID
                      <SortIcon column="customer_id" />
                    </div>
                  </th>
                  <th
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('score')}
                  >
                    <div className="flex items-center">
                      Credit Score
                      <SortIcon column="score" />
                    </div>
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Score Label
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Risk Level
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Probability
                  </th>
                  <th
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('date')}
                  >
                    <div className="flex items-center">
                      Last Updated
                      <SortIcon column="date" />
                    </div>
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Model Version
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {scores.map((score) => (
                  <tr key={score.customer_id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <User className="w-4 h-4 text-gray-400" />
                        <span className="text-sm font-medium text-gray-900">
                          {score.customer_id}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <span className={`text-2xl font-bold ${getScoreColor(score.customer_score)}`}>
                          {score.customer_score}
                        </span>
                        <span className="text-xs text-gray-500">/100</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border ${getScoreBg(score.customer_score)}`}>
                        {getScoreLabel(score.customer_score)}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getRiskBadgeClass(score.risk_level)}`}>
                        {score.risk_level?.toUpperCase() || 'UNKNOWN'}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                      {(score.probability * 100).toFixed(2)}%
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(score.created_at)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                      {score.model_version || 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Summary Footer */}
          <div className="p-4 border-t border-gray-200">
            <div className="text-sm text-gray-600">
              Showing <span className="font-medium">{scores.length}</span> unique customers with scores
              {allPredictionsLoaded && (
                <span className="ml-2 text-xs text-gray-500">
                  (from {total} total predictions)
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CustomerScoresTable;
