import React, { createContext, useContext, useState, useEffect } from 'react';
import { creditScoringAPI } from '../utils/api';

const AuthContext = createContext(null);

function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const logout = async () => {
    try {
      await creditScoringAPI.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_data');
      setUser(null);
      setIsAuthenticated(false);
    }
  };

  // Check if user is logged in on mount
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    const savedUser = localStorage.getItem('user_data');
    
    if (token && savedUser) {
      try {
        const userData = JSON.parse(savedUser);
        setUser(userData);
        setIsAuthenticated(true);
        // Verify token is still valid
        creditScoringAPI.getCurrentUser()
          .then((currentUser) => {
            setUser(currentUser);
            localStorage.setItem('user_data', JSON.stringify(currentUser));
          })
          .catch(() => {
            // Token invalid, clear everything
            localStorage.removeItem('auth_token');
            localStorage.removeItem('user_data');
            setUser(null);
            setIsAuthenticated(false);
          })
          .finally(() => setLoading(false));
      } catch (error) {
        console.error('Error parsing user data:', error);
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        setUser(null);
        setIsAuthenticated(false);
        setLoading(false);
      }
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (username, password) => {
    try {
      const response = await creditScoringAPI.login(username, password);
      const { access_token, user: userData } = response;
      
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user_data', JSON.stringify(userData));
      
      setUser(userData);
      setIsAuthenticated(true);
      
      return { success: true, user: userData };
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Login failed. Please check your credentials.',
      };
    }
  };

  const hasPermission = (permission) => {
    if (!user) return false;
    if (user.is_superuser) return true;
    return user.permissions?.includes(permission) || false;
  };

  const hasRole = (roleCode) => {
    if (!user) return false;
    if (user.is_superuser) return true;
    return user.roles?.some(role => role.role_code === roleCode) || false;
  };

  const hasAnyRole = (roleCodes) => {
    if (!user) return false;
    if (user.is_superuser) return true;
    return roleCodes.some(roleCode => hasRole(roleCode));
  };

  const value = {
    user,
    loading,
    isAuthenticated,
    login,
    logout,
    hasPermission,
    hasRole,
    hasAnyRole,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Export hook separately for Fast Refresh compatibility
export { useAuth };
