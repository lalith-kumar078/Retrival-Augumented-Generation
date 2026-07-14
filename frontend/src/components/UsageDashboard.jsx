import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { getUsageStats } from '../services/api';

export default function UsageDashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const data = await getUsageStats();
      setStats(data);
      setError(null);
    } catch (err) {
      setError('Failed to load usage statistics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <main className="main-content usage-dashboard">
        <div className="chat-header"><h2>Usage Dashboard</h2></div>
        <div className="loading-state">Loading usage stats...</div>
      </main>
    );
  }

  if (error || !stats) {
    return (
      <main className="main-content usage-dashboard">
        <div className="chat-header"><h2>Usage Dashboard</h2></div>
        <div className="error-state">{error || 'No data available'}</div>
      </main>
    );
  }

  const { total_tokens_today, daily_limit, timeseries, total_requests_today } = stats;

  const tokensRemaining = Math.max(0, daily_limit - total_tokens_today);
  const pieData = [
    { name: 'Tokens Used', value: total_tokens_today },
    { name: 'Tokens Remaining', value: tokensRemaining }
  ];
  
  const COLORS = ['var(--accent)', 'var(--bg-hover)'];

  // Calculate average tokens per request
  const avgTokens = total_requests_today > 0 
    ? Math.round(total_tokens_today / total_requests_today) 
    : 0;

  // Calculate requests remaining at current rate
  const requestsRemaining = avgTokens > 0 
    ? Math.floor(tokensRemaining / avgTokens) 
    : 0;

  // Format time for chart
  const formattedTimeseries = timeseries.map(item => ({
    ...item,
    time: new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }));

  return (
    <main className="main-content usage-dashboard">
      <div className="chat-header">
        <div className="chat-header-left">
          <h2>Usage Dashboard</h2>
          <span className="chat-doc-scope">Tokens & Requests</span>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="stats-cards">
          <div className="stat-card">
            <h4>Total Tokens Today</h4>
            <div className="stat-value">{total_tokens_today.toLocaleString()}</div>
          </div>
          <div className="stat-card">
            <h4>Average Tokens / Query</h4>
            <div className="stat-value">{avgTokens.toLocaleString()}</div>
          </div>
          <div className="stat-card">
            <h4>Requests Remaining</h4>
            <div className="stat-value">~{requestsRemaining.toLocaleString()}</div>
          </div>
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <h3>Daily Token Quota</h3>
            <div className="chart-container pie-container">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    fill="#8884d8"
                    paddingAngle={5}
                    dataKey="value"
                    stroke="none"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'var(--bg-panel)', border: '0.5px solid var(--border)', borderRadius: '6px' }}
                    itemStyle={{ color: 'var(--text-primary)' }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="pie-legend">
                <div className="legend-item">
                  <span className="legend-color" style={{ backgroundColor: COLORS[0] }}></span>
                  <span className="legend-label">Used ({total_tokens_today.toLocaleString()})</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color" style={{ backgroundColor: COLORS[1] }}></span>
                  <span className="legend-label">Remaining ({tokensRemaining.toLocaleString()})</span>
                </div>
              </div>
            </div>
          </div>

          <div className="chart-card">
            <h3>Usage Rate (Last 50 Requests)</h3>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={formattedTimeseries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis 
                    dataKey="time" 
                    stroke="var(--text-muted)" 
                    fontSize={10} 
                    tickMargin={10}
                    tickFormatter={(val, i) => i % 5 === 0 ? val : ''}
                  />
                  <YAxis 
                    stroke="var(--text-muted)" 
                    fontSize={10} 
                    tickFormatter={(val) => `${val / 1000}k`}
                  />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'var(--bg-panel)', border: '0.5px solid var(--border)', borderRadius: '6px', fontSize: '12px' }}
                    labelStyle={{ color: 'var(--text-secondary)', marginBottom: '4px' }}
                    itemStyle={{ color: 'var(--accent)' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="tokens" 
                    stroke="var(--accent)" 
                    strokeWidth={2}
                    dot={{ fill: 'var(--accent)', r: 2 }}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
