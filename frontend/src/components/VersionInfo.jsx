import React, { useState, useEffect } from 'react';
import { GitBranch, Database, Package, Clock, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const VersionInfo = () => {
  const [versions, setVersions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadVersions();
  }, []);

  const loadVersions = async () => {
    try {
      const data = await creditScoringAPI.getVersions();
      setVersions(data);
    } catch (error) {
      console.error('Failed to load versions:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="card animate-fade-in">
        <div className="text-center py-12">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-600">Loading version information...</p>
        </div>
      </div>
    );
  }

  if (!versions) {
    return (
      <div className="card animate-fade-in">
        <div className="text-center py-12">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-slate-600">Failed to load version information</p>
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Info },
    { id: 'model', label: 'Model Versions', icon: Package },
    { id: 'data', label: 'Data Versions', icon: Database },
  ];

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    try {
      // Handle both ISO strings and Unix timestamps
      const date = typeof timestamp === 'number' 
        ? new Date(timestamp / 1000) 
        : new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  };

  return (
    <div className="card animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-purple-100 rounded-lg">
          <GitBranch className="w-6 h-6 text-purple-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Version Information</h2>
          <p className="text-slate-600 text-sm">Model, data, and system version tracking</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-slate-200 mb-6">
        <div className="flex gap-2 overflow-x-auto">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`px-4 py-2 font-semibold transition-colors whitespace-nowrap flex items-center gap-2 ${
                activeTab === id
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6 animate-fade-in">
          {/* Current Production Versions */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-6 border border-green-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              Current Production Versions
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Model Version */}
              <div className="bg-white rounded-lg p-4 border border-green-200">
                <div className="flex items-center gap-2 mb-2">
                  <Package className="w-5 h-5 text-purple-600" />
                  <span className="font-semibold text-slate-800">Model</span>
                </div>
                {versions.model_versions?.current ? (
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-slate-600">Version:</span>
                      <span className="font-bold text-slate-800">
                        v{versions.model_versions.current.version}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">Stage:</span>
                      <span className="font-semibold text-green-600">
                        {versions.model_versions.current.stage}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">Created:</span>
                      <span className="text-slate-700">
                        {formatTimestamp(versions.model_versions.current.created_at)}
                      </span>
                    </div>
                    {versions.model_versions.current.metrics?.test_roc_auc && (
                      <div className="flex justify-between">
                        <span className="text-slate-600">ROC-AUC:</span>
                        <span className="font-semibold text-slate-800">
                          {versions.model_versions.current.metrics.test_roc_auc.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-slate-600">No production model found</p>
                )}
              </div>

              {/* Data Versions */}
              <div className="bg-white rounded-lg p-4 border border-green-200">
                <div className="flex items-center gap-2 mb-2">
                  <Database className="w-5 h-5 text-blue-600" />
                  <span className="font-semibold text-slate-800">Data</span>
                </div>
                {versions.data_versions && Object.keys(versions.data_versions).length > 0 ? (
                  <div className="space-y-1 text-sm">
                    {Object.entries(versions.data_versions).slice(0, 3).map(([type, versions_obj]) => {
                      if (typeof versions_obj === 'object' && versions_obj !== null) {
                        const latest = Object.values(versions_obj).sort((a, b) => 
                          new Date(b.created || 0) - new Date(a.created || 0)
                        )[0];
                        if (latest) {
                          return (
                            <div key={type} className="flex justify-between">
                              <span className="text-slate-600 capitalize">{type}:</span>
                              <span className="font-semibold text-slate-800">
                                {latest.version}
                              </span>
                            </div>
                          );
                        }
                      }
                      return null;
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-slate-600">No data versions found</p>
                )}
              </div>
            </div>
          </div>

          {/* System Information */}
          <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4">System Information</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
              <div>
                <div className="text-slate-600 mb-1">Python Version</div>
                <div className="font-semibold text-slate-800">
                  {versions.python_version || 'N/A'}
                </div>
              </div>
              <div>
                <div className="text-slate-600 mb-1">Last Updated</div>
                <div className="font-semibold text-slate-800">
                  {formatDate(versions.timestamp)}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Versions Tab */}
      {activeTab === 'model' && (
        <div className="space-y-6 animate-fade-in">
          {versions.model_versions?.all_versions && versions.model_versions.all_versions.length > 0 ? (
            <div className="space-y-4">
              {versions.model_versions.all_versions.map((version, idx) => (
                <div
                  key={idx}
                  className="bg-white rounded-lg p-6 border-2 border-slate-200 hover:border-purple-300 transition-all"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        version.stage === 'Production' ? 'bg-green-100 text-green-700' :
                        version.stage === 'Staging' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-slate-100 text-slate-700'
                      }`}>
                        {version.stage || 'None'}
                      </div>
                      <span className="text-xl font-bold text-slate-800">
                        Version {version.version}
                      </span>
                    </div>
                    <div className="text-sm text-slate-600">
                      {formatTimestamp(version.created_at)}
                    </div>
                  </div>

                  {version.metrics && Object.keys(version.metrics).length > 0 && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      {Object.entries(version.metrics).slice(0, 4).map(([metric, value]) => (
                        <div key={metric} className="bg-slate-50 rounded-lg p-3">
                          <div className="text-xs text-slate-600 mb-1 uppercase tracking-wide">
                            {metric.replace('_', ' ')}
                          </div>
                          <div className="text-lg font-bold text-slate-800">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {version.description && (
                    <div className="text-sm text-slate-600 italic">
                      {version.description}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-slate-50 rounded-lg border border-slate-200">
              <Package className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-600">No model versions found</p>
            </div>
          )}
        </div>
      )}

      {/* Data Versions Tab */}
      {activeTab === 'data' && (
        <div className="space-y-6 animate-fade-in">
          {versions.data_versions && Object.keys(versions.data_versions).length > 0 ? (
            <div className="space-y-4">
              {Object.entries(versions.data_versions).map(([dataType, versions_obj]) => (
                <div key={dataType} className="bg-white rounded-lg p-6 border-2 border-slate-200">
                  <h3 className="text-lg font-bold text-slate-800 mb-4 capitalize flex items-center gap-2">
                    <Database className="w-5 h-5 text-blue-600" />
                    {dataType.replace('_', ' ')}
                  </h3>
                  
                  {typeof versions_obj === 'object' && versions_obj !== null ? (
                    <div className="space-y-3">
                      {Object.entries(versions_obj)
                        .sort((a, b) => {
                          const aVer = a[0].replace('v', '');
                          const bVer = b[0].replace('v', '');
                          return parseInt(bVer) - parseInt(aVer);
                        })
                        .slice(0, 5)
                        .map(([version, info]) => (
                          <div
                            key={version}
                            className="bg-slate-50 rounded-lg p-4 border border-slate-200"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-bold text-slate-800">{version}</span>
                              <span className="text-xs text-slate-600 flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {formatDate(info.created)}
                              </span>
                            </div>
                            {info.metadata && Object.keys(info.metadata).length > 0 && (
                              <div className="text-xs text-slate-600 mt-2">
                                {Object.entries(info.metadata).slice(0, 3).map(([key, value]) => (
                                  <span key={key} className="mr-3">
                                    <strong>{key}:</strong> {String(value)}
                                  </span>
                                ))}
                              </div>
                            )}
                            {info.checksum && (
                              <div className="text-xs text-slate-500 mt-2 font-mono">
                                Checksum: {info.checksum.substring(0, 16)}...
                              </div>
                            )}
                          </div>
                        ))}
                    </div>
                  ) : (
                    <p className="text-slate-600 text-sm">No versions available</p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-slate-50 rounded-lg border border-slate-200">
              <Database className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-600">No data versions found</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VersionInfo;
