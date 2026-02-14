import React, { useState, useEffect } from 'react';
import { GitBranch, Database, Package, Target, ArrowRight, Search, Filter, RefreshCw, Info, ExternalLink } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const DataLineage = () => {
  const [lineage, setLineage] = useState([]);
  const [dataVersions, setDataVersions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedDataVersion, setSelectedDataVersion] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [viewMode, setViewMode] = useState('graph'); // 'graph' or 'table'
  const [filterType, setFilterType] = useState('all'); // 'all', 'prediction', 'model', 'feature_set'

  useEffect(() => {
    loadDataVersions();
    loadLineage();
  }, []);

  const loadDataVersions = async () => {
    try {
      const data = await creditScoringAPI.getDataVersions();
      // Support both old format (object) and new format (array)
      let versions = [];
      
      if (data.all_versions) {
        if (data.all_versions.raw_transactions) {
          if (Array.isArray(data.all_versions.raw_transactions)) {
            versions = data.all_versions.raw_transactions;
          } else {
            versions = Object.values(data.all_versions.raw_transactions);
          }
        }
      }
      
      // Sort by version number (newest first)
      const sortedVersions = versions.sort((a, b) => {
        const aNum = parseInt(a.version?.replace('v', '') || '0');
        const bNum = parseInt(b.version?.replace('v', '') || '0');
        return bNum - aNum;
      });
      
      setDataVersions(sortedVersions);
    } catch (err) {
      console.error('Failed to load data versions:', err);
      setError('Failed to load data versions');
    }
  };

  const loadLineage = async (dataVersionId = null, targetType = null, targetId = null) => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getLineage(dataVersionId, targetType, targetId);
      setLineage(data.lineage || []);
    } catch (err) {
      setError(err.message || 'Failed to load lineage data');
      console.error('Error loading lineage:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadLineageByDataVersion = async (dataVersionId) => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getLineageByDataVersion(dataVersionId);
      setSelectedDataVersion(data.data_version);
      setLineage(data.downstream || []);
    } catch (err) {
      setError(err.message || 'Failed to load lineage');
      console.error('Error loading lineage:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadLineageByTarget = async (targetType, targetId) => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getLineageByTarget(targetType, targetId);
      setSelectedTarget({ type: targetType, id: targetId });
      setLineage(data.upstream || []);
    } catch (err) {
      setError(err.message || 'Failed to load lineage');
      console.error('Error loading lineage:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDataVersionSelect = async (version) => {
    setSelectedDataVersion(version);
    setSelectedTarget(null);
    
    // If version has an ID, load lineage by data version ID
    if (version && version.id) {
      await loadLineageByDataVersion(version.id);
    } else {
      // Otherwise, load all lineage
      await loadLineage(null, null, null);
    }
  };

  const handleTargetClick = (targetType, targetId) => {
    loadLineageByTarget(targetType, targetId);
  };

  const filteredLineage = lineage.filter(item => {
    if (filterType === 'all') return true;
    return item.target?.type === filterType;
  });

  const getTargetIcon = (targetType) => {
    switch (targetType) {
      case 'model':
        return Package;
      case 'prediction':
        return Target;
      case 'feature_set':
        return Database;
      default:
        return Database;
    }
  };

  const getTargetColor = (targetType) => {
    switch (targetType) {
      case 'model':
        return 'text-purple-600 bg-purple-100 border-purple-200';
      case 'prediction':
        return 'text-blue-600 bg-blue-100 border-blue-200';
      case 'feature_set':
        return 'text-green-600 bg-green-100 border-green-200';
      default:
        return 'text-gray-600 bg-gray-100 border-gray-200';
    }
  };

  const groupedBySource = filteredLineage.reduce((acc, item) => {
    const key = `${item.source.data_type}:${item.source.version}`;
    if (!acc[key]) {
      acc[key] = {
        source: item.source,
        targets: []
      };
    }
    acc[key].targets.push(item.target);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-blue-600" />
            Data Lineage
          </h2>
          <p className="text-sm text-gray-600 mt-1">Track data flow from versions to models and predictions</p>
        </div>
        <button
          onClick={() => loadLineage()}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Filters and Controls */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Data Version Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Data Version
            </label>
            <select
              value={selectedDataVersion?.version || ''}
              onChange={async (e) => {
                if (e.target.value === '') {
                  setSelectedDataVersion(null);
                  setSelectedTarget(null);
                  await loadLineage();
                } else {
                  const version = dataVersions.find(v => v.version === e.target.value);
                  if (version) {
                    await handleDataVersionSelect(version);
                  }
                }
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">All Versions</option>
              {dataVersions.map((version) => (
                <option key={version.version || version.id} value={version.version}>
                  {version.version} - {version.created ? new Date(version.created).toLocaleDateString() : 'Unknown date'}
                  {version.id ? ` (ID: ${version.id})` : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Target Type Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Type
            </label>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Types</option>
              <option value="prediction">Predictions</option>
              <option value="model">Models</option>
              <option value="feature_set">Feature Sets</option>
            </select>
          </div>

          {/* View Mode */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              View Mode
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setViewMode('graph')}
                className={`flex-1 px-3 py-2 rounded-lg transition-colors ${
                  viewMode === 'graph'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Graph
              </button>
              <button
                onClick={() => setViewMode('table')}
                className={`flex-1 px-3 py-2 rounded-lg transition-colors ${
                  viewMode === 'table'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Table
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading lineage data...</p>
        </div>
      )}

      {/* Lineage Visualization */}
      {!loading && !error && (
        <>
          {viewMode === 'graph' ? (
            /* Graph View */
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              {filteredLineage.length === 0 ? (
                <div className="text-center py-12">
                  <Info className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">No lineage data found</p>
                  <p className="text-sm text-gray-500 mt-2">
                    Upload data and make predictions to see lineage relationships
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {Object.values(groupedBySource).map((group, idx) => (
                    <div key={idx} className="border-l-4 border-blue-500 pl-4">
                      {/* Source */}
                      <div className="flex items-center gap-3 mb-4">
                        <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
                          <Database className="w-5 h-5 text-blue-600" />
                          <div>
                            <p className="font-semibold text-blue-900">
                              {group.source.data_type}: {group.source.version}
                            </p>
                            <p className="text-xs text-blue-600">
                              Checksum: {group.source.checksum || 'N/A'}
                            </p>
                          </div>
                        </div>
                        <ArrowRight className="w-5 h-5 text-gray-400" />
                        <span className="text-sm text-gray-500">
                          {group.targets.length} downstream item{group.targets.length !== 1 ? 's' : ''}
                        </span>
                      </div>

                      {/* Targets */}
                      <div className="ml-8 space-y-3">
                        {group.targets.map((target, tIdx) => {
                          const TargetIcon = getTargetIcon(target.type);
                          const colorClass = getTargetColor(target.type);
                          return (
                            <div
                              key={tIdx}
                              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
                              onClick={() => handleTargetClick(target.type, target.id)}
                            >
                              <div className={`flex items-center gap-2 px-4 py-2 border rounded-lg ${colorClass}`}>
                                <TargetIcon className="w-4 h-4" />
                                <div>
                                  <p className="font-medium text-sm">{target.name || target.id}</p>
                                  <p className="text-xs opacity-75">{target.type}</p>
                                </div>
                              </div>
                              <ExternalLink className="w-4 h-4 text-gray-400" />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            /* Table View */
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Source Data Version
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Target Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Target ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Relationship
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Created
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredLineage.length === 0 ? (
                      <tr>
                        <td colSpan="6" className="px-6 py-12 text-center text-gray-500">
                          No lineage data found
                        </td>
                      </tr>
                    ) : (
                      filteredLineage.map((item) => {
                        const TargetIcon = getTargetIcon(item.target.type);
                        return (
                          <tr key={item.id} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center gap-2">
                                <Database className="w-4 h-4 text-blue-600" />
                                <div>
                                  <p className="text-sm font-medium text-gray-900">
                                    {item.source.data_type}: {item.source.version}
                                  </p>
                                  <p className="text-xs text-gray-500">
                                    {item.source.checksum?.substring(0, 16)}...
                                  </p>
                                </div>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center gap-2">
                                <TargetIcon className="w-4 h-4" />
                                <span className="text-sm text-gray-900">{item.target.type}</span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <p className="text-sm text-gray-900">{item.target.name || item.target.id}</p>
                              <p className="text-xs text-gray-500">{item.target.id}</p>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded">
                                {item.relationship.type}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {item.created_at ? new Date(item.created_at).toLocaleString() : 'N/A'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                              <button
                                onClick={() => handleTargetClick(item.target.type, item.target.id)}
                                className="text-blue-600 hover:text-blue-800 flex items-center gap-1"
                              >
                                View Upstream
                                <ExternalLink className="w-3 h-3" />
                              </button>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Selected Target Info */}
      {selectedTarget && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-semibold text-blue-900">
                Viewing upstream for: {selectedTarget.type} - {selectedTarget.id}
              </p>
              <p className="text-sm text-blue-700 mt-1">
                Showing {lineage.length} data version{lineage.length !== 1 ? 's' : ''} used
              </p>
            </div>
            <button
              onClick={() => {
                setSelectedTarget(null);
                loadLineage();
              }}
              className="text-blue-600 hover:text-blue-800 text-sm"
            >
              Clear Selection
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataLineage;
