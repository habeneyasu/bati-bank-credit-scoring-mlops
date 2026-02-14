import React, { useState } from 'react';
import { Upload, File, CheckCircle, AlertCircle, Loader, X, Database } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const DataUpload = () => {
  const [file, setFile] = useState(null);
  const [dataSource, setDataSource] = useState('manual_upload');
  const [dataVersion, setDataVersion] = useState('');
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (selectedFile) => {
    const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
    if (fileExtension !== 'csv' && fileExtension !== 'json') {
      setError('Please select a CSV or JSON file');
      return;
    }
    setFile(selectedFile);
    setError(null);
    setResult(null);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('data_source', dataSource);
      if (dataVersion) {
        formData.append('data_version', dataVersion);
      }

      const response = await creditScoringAPI.uploadRawData(formData);
      setResult(response);
      setFile(null);
      
      // Reset file input
      const fileInput = document.getElementById('file-input');
      if (fileInput) {
        fileInput.value = '';
      }
      
      // Trigger a custom event to notify other components
      window.dispatchEvent(new CustomEvent('dataUploaded', { detail: response }));
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Upload failed');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setDataSource('manual_upload');
    setDataVersion('');
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-blue-100 rounded-lg">
            <Database className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Upload Raw Data</h2>
            <p className="text-gray-600 text-sm">Upload transaction data from CSV or JSON files</p>
          </div>
        </div>

        {/* Upload Form */}
        <div className="space-y-4">
          {/* File Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-blue-500 bg-blue-50'
                : file
                ? 'border-green-500 bg-green-50'
                : 'border-gray-300 bg-gray-50 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {file ? (
              <div className="space-y-2">
                <CheckCircle className="w-12 h-12 text-green-500 mx-auto" />
                <div>
                  <p className="font-medium text-gray-900">{file.name}</p>
                  <p className="text-sm text-gray-500">
                    {(file.size / 1024).toFixed(2)} KB
                  </p>
                </div>
                <button
                  onClick={() => setFile(null)}
                  className="text-sm text-red-600 hover:text-red-700 mt-2"
                >
                  Remove file
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                <div>
                  <label
                    htmlFor="file-input"
                    className="cursor-pointer text-blue-600 hover:text-blue-700 font-medium"
                  >
                    Click to upload
                  </label>
                  <span className="text-gray-500"> or drag and drop</span>
                </div>
                <p className="text-sm text-gray-500">CSV or JSON files only</p>
                <input
                  id="file-input"
                  type="file"
                  accept=".csv,.json"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>
            )}
          </div>

          {/* Data Source */}
          <div>
            <label htmlFor="data-source" className="block text-sm font-medium text-gray-700 mb-2">
              Data Source
            </label>
            <input
              id="data-source"
              type="text"
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value)}
              placeholder="e.g., ecommerce_platform, payment_gateway"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Data Version (Optional) */}
          <div>
            <label htmlFor="data-version" className="block text-sm font-medium text-gray-700 mb-2">
              Data Version <span className="text-gray-400">(Optional)</span>
            </label>
            <input
              id="data-version"
              type="text"
              value={dataVersion}
              onChange={(e) => setDataVersion(e.target.value)}
              placeholder="e.g., v1.0, 2024-02"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-red-800">Upload Failed</p>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-600 hover:text-red-700"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Success Result */}
          {result && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-green-800">Upload Successful!</p>
                  <div className="mt-2 space-y-1 text-sm text-green-700">
                    <p>• Uploaded: {result.uploaded_count} transactions</p>
                    <p>• Total rows: {result.total_rows}</p>
                    {result.validation_errors > 0 && (
                      <p className="text-yellow-700">
                        • Validation errors: {result.validation_errors}
                      </p>
                    )}
                    <p>• File: {result.file_name}</p>
                    <p>• Source: {result.data_source}</p>
                  </div>
                </div>
                <button
                  onClick={handleReset}
                  className="text-green-600 hover:text-green-700"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center gap-3 pt-4">
            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="flex-1 bg-blue-600 text-white px-6 py-2.5 rounded-lg font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {uploading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5" />
                  Upload Data
                </>
              )}
            </button>
            {(file || result) && (
              <button
                onClick={handleReset}
                className="px-6 py-2.5 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
              >
                Reset
              </button>
            )}
          </div>
        </div>
      </div>

      {/* File Format Guide */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-blue-900 mb-2 flex items-center gap-2">
          <File className="w-4 h-4" />
          File Format Requirements
        </h3>
        <div className="text-sm text-blue-800 space-y-1">
          <p><strong>CSV Format:</strong> First row must contain headers (transaction_id, customer_id, amount, transaction_start_time, etc.)</p>
          <p><strong>JSON Format:</strong> Array of transaction objects or object with 'transactions' key</p>
          <p><strong>Required Fields:</strong> transaction_id, customer_id, amount, transaction_start_time</p>
          <p><strong>Optional Fields:</strong> batch_id, account_id, currency_code, product_category, channel_id, etc.</p>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;
