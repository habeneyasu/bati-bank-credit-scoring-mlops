import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Search } from 'lucide-react';

const FeatureInputForm = ({ featureNames, features, onChange }) => {
  const [expandedSections, setExpandedSections] = useState({});
  const [searchTerm, setSearchTerm] = useState('');

  const groupedFeatures = React.useMemo(() => {
    const groups = {
      'RFM Metrics': [],
      'Transaction Patterns': [],
      'Temporal Features': [],
      'Customer Segments': [],
      'WoE Features': [],
      'Other Features': [],
    };

    featureNames.forEach((name, idx) => {
      const lower = name.toLowerCase();
      if (lower.includes('rfm')) {
        groups['RFM Metrics'].push({ name, idx });
      } else if (lower.includes('transaction') || lower.includes('spend') || lower.includes('amount')) {
        groups['Transaction Patterns'].push({ name, idx });
      } else if (lower.includes('hour') || lower.includes('day') || lower.includes('weekend') || lower.includes('evening')) {
        groups['Temporal Features'].push({ name, idx });
      } else if (lower.includes('cluster')) {
        groups['Customer Segments'].push({ name, idx });
      } else if (lower.includes('woe')) {
        groups['WoE Features'].push({ name, idx });
      } else {
        groups['Other Features'].push({ name, idx });
      }
    });

    // Filter out empty groups and apply search
    const filtered = {};
    Object.entries(groups).forEach(([key, items]) => {
      if (items.length > 0) {
        const filteredItems = items.filter(item =>
          item.name.toLowerCase().includes(searchTerm.toLowerCase())
        );
        if (filteredItems.length > 0) {
          filtered[key] = filteredItems;
        }
      }
    });

    return filtered;
  }, [featureNames, searchTerm]);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const handleFeatureChange = (idx, value) => {
    onChange({
      ...features,
      [idx]: parseFloat(value) || 0,
    });
  };

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
        <input
          type="text"
          placeholder="Search features..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="input-field pl-10"
        />
      </div>

      {/* Feature Groups */}
      <div className="max-h-[600px] overflow-y-auto space-y-3 pr-2">
        {Object.entries(groupedFeatures).map(([sectionName, items]) => (
          <div key={sectionName} className="border border-slate-200 rounded-lg overflow-hidden">
            <button
              onClick={() => toggleSection(sectionName)}
              className="w-full px-4 py-3 bg-gradient-to-r from-slate-50 to-slate-100 hover:from-slate-100 hover:to-slate-200 transition-all flex items-center justify-between font-semibold text-slate-700"
            >
              <span>{sectionName}</span>
              <span className="text-xs text-slate-500 bg-white px-2 py-1 rounded-full">
                {items.length}
              </span>
              {expandedSections[sectionName] ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {expandedSections[sectionName] && (
              <div className="p-4 space-y-3 bg-white">
                {items.map(({ name, idx }) => (
                  <div key={idx} className="space-y-1">
                    <label className="text-sm font-medium text-slate-700 block">
                      {name}
                    </label>
                    <input
                      type="number"
                      step="0.0001"
                      value={features[idx] || 0}
                      onChange={(e) => handleFeatureChange(idx, e.target.value)}
                      className="input-field text-sm"
                      placeholder="0.0"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="pt-4 border-t border-slate-200">
        <button
          onClick={() => {
            const allExpanded = {};
            Object.keys(groupedFeatures).forEach(key => {
              allExpanded[key] = true;
            });
            setExpandedSections(allExpanded);
          }}
          className="text-sm text-blue-600 hover:text-blue-700 font-medium"
        >
          Expand All
        </button>
        <span className="mx-2 text-slate-300">|</span>
        <button
          onClick={() => setExpandedSections({})}
          className="text-sm text-blue-600 hover:text-blue-700 font-medium"
        >
          Collapse All
        </button>
      </div>
    </div>
  );
};

export default FeatureInputForm;
