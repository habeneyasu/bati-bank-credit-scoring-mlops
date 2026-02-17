import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { LogIn, Lock, User, AlertCircle, Loader, CheckCircle, XCircle } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking'); // 'checking', 'connected', 'disconnected'
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    // Check API connection status
    const checkConnection = async () => {
      try {
        await creditScoringAPI.healthCheck();
        setApiStatus('connected');
      } catch (error) {
        setApiStatus('disconnected');
      }
    };
    checkConnection();
    // Check every 5 seconds
    const interval = setInterval(checkConnection, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const result = await login(username, password);

    if (result.success) {
      // Determine redirect based on user role
      const user = result.user;
      let redirectPath = '/dashboard';
      
      // Role-based redirects
      if (user.is_superuser || user.roles?.some(r => r.role_code === 'super_admin')) {
        redirectPath = '/dashboard';
      } else if (user.roles?.some(r => r.role_code === 'data_admin')) {
        redirectPath = '/dashboard';
      } else if (user.roles?.some(r => r.role_code === 'model_developer')) {
        redirectPath = '/dashboard';
      } else if (user.roles?.some(r => r.role_code === 'data_analyst')) {
        redirectPath = '/dashboard';
      } else if (user.roles?.some(r => r.role_code === 'business_user')) {
        redirectPath = '/dashboard';
      } else {
        redirectPath = '/dashboard';
      }
      
      navigate(redirectPath);
    } else {
      const errorMessage = result.error || 'Login failed. Please try again.';
      setError(errorMessage);
      setLoading(false);
      
      // If it's a connection error, update API status
      if (errorMessage.includes('Network') || errorMessage.includes('connection') || errorMessage.includes('fetch')) {
        setApiStatus('disconnected');
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        {/* Logo/Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl shadow-lg mb-4">
            <Lock className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Bati Bank</h1>
          <p className="text-gray-600">Credit Scoring MLOps Platform</p>
          
          {/* API Connection Status */}
          <div className="mt-4 flex items-center justify-center gap-2">
            {apiStatus === 'checking' && (
              <div className="flex items-center gap-2 text-gray-500 text-sm">
                <Loader className="w-4 h-4 animate-spin" />
                <span>Checking connection...</span>
              </div>
            )}
            {apiStatus === 'connected' && (
              <div className="flex items-center gap-2 text-green-600 text-sm">
                <CheckCircle className="w-4 h-4" />
                <span>API Connected</span>
              </div>
            )}
            {apiStatus === 'disconnected' && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <XCircle className="w-4 h-4" />
                <span>API Disconnected</span>
              </div>
            )}
          </div>
        </div>

        {/* Login Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <div className="mb-6">
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">Sign In</h2>
            <p className="text-gray-600 text-sm">Enter your credentials to access the platform</p>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Username Field */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                Username
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  className="block w-full pl-10 pr-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                  placeholder="Enter your username"
                  disabled={loading}
                />
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="block w-full pl-10 pr-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                  placeholder="Enter your password"
                  disabled={loading}
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-2.5 px-4 rounded-lg font-medium hover:from-blue-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Signing in...
                </>
              ) : (
                <>
                  <LogIn className="w-5 h-5" />
                  Sign In
                </>
              )}
            </button>
          </form>

          {/* Demo Users Info */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <p className="text-xs font-semibold text-gray-700 mb-3">Demo Credentials:</p>
            <div className="space-y-2 text-xs">
              <button
                type="button"
                onClick={() => {
                  setUsername('data_admin');
                  setPassword('DataAdmin@2024');
                }}
                className="w-full text-left p-2 rounded-lg hover:bg-gray-50 border border-gray-200 transition-colors"
              >
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 font-medium">Data Administrator</span>
                  <span className="text-blue-600 font-mono text-[10px]">Click to fill</span>
              </div>
              </button>
              <button
                type="button"
                onClick={() => {
                  setUsername('data_analyst');
                  setPassword('Analyst@2024');
                }}
                className="w-full text-left p-2 rounded-lg hover:bg-gray-50 border border-gray-200 transition-colors"
              >
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 font-medium">Data Analyst</span>
                  <span className="text-blue-600 font-mono text-[10px]">Click to fill</span>
              </div>
              </button>
              <button
                type="button"
                onClick={() => {
                  setUsername('business_user');
                  setPassword('Business@2024');
                }}
                className="w-full text-left p-2 rounded-lg hover:bg-gray-50 border border-gray-200 transition-colors"
              >
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 font-medium">Business User</span>
                  <span className="text-blue-600 font-mono text-[10px]">Click to fill</span>
              </div>
              </button>
              <button
                type="button"
                onClick={() => {
                  setUsername('model_dev');
                  setPassword('ModelDev@2024');
                }}
                className="w-full text-left p-2 rounded-lg hover:bg-gray-50 border border-gray-200 transition-colors"
              >
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 font-medium">Model Developer</span>
                  <span className="text-blue-600 font-mono text-[10px]">Click to fill</span>
              </div>
              </button>
            </div>
            <p className="text-[10px] text-gray-500 mt-3 text-center">
              Click any role above to auto-fill credentials
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-gray-500 mt-6">
          © 2024 Bati Bank. All rights reserved.
        </p>
      </div>
    </div>
  );
};

export default Login;
