import React, { useEffect, useState } from 'react';

export default function Toast({ toasts, removeToast }) {
  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onRemove }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onRemove(toast.id);
    }, 4000);
    return () => clearTimeout(timer);
  }, [toast.id, onRemove]);

  const icons = {
    success: '✓',
    error: '✕',
    info: 'ℹ',
  };

  return (
    <div className={`toast ${toast.type}`}>
      <span>{icons[toast.type] || 'ℹ'}</span>
      {toast.message}
    </div>
  );
}
