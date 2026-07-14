import React, { useRef, useState, useEffect } from 'react';
import { uploadDocument, deleteDocument, getDocumentSessions, getDocument } from '../services/api';

export default function Sidebar({
  documents,
  onRefreshDocs,
  onToast,
  health,
  theme,
  onToggleTheme,
  onDocumentClick,
  activeDocumentId,
  activeSessionId,
  onSelectSession,
  onCreateSessionForDoc,
  onDeleteSession,
  isStreaming,
  view,
  setView,
}) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [expandedDocId, setExpandedDocId] = useState(null);
  const [docSessions, setDocSessions] = useState([]);
  const [loadingSessions, setLoadingSessions] = useState(false);

  const fetchDocSessions = async (docId) => {
    setLoadingSessions(true);
    try {
      const sessionsList = await getDocumentSessions(docId);
      setDocSessions(sessionsList);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingSessions(false);
    }
  };

  useEffect(() => {
    if (activeDocumentId) {
      setExpandedDocId(activeDocumentId);
      fetchDocSessions(activeDocumentId);
    }
  }, [activeDocumentId, activeSessionId, isStreaming]);

  const handleDocumentClickLocal = async (doc) => {
    if (expandedDocId === doc.doc_id) {
      setExpandedDocId(null);
      setDocSessions([]);
    } else {
      setExpandedDocId(doc.doc_id);
      setDocSessions([]);
      setLoadingSessions(true);
      try {
        const sessionsList = await getDocumentSessions(doc.doc_id);
        setDocSessions(sessionsList);
      } catch (err) {
        console.error(err);
        onToast?.('Failed to fetch sessions for document', 'error');
      } finally {
        setLoadingSessions(false);
      }
    }
    onDocumentClick?.(doc);
  };

  const handleDeleteSessionLocal = async (e, sessionId) => {
    e.stopPropagation();
    try {
      await onDeleteSession?.(sessionId);
      setDocSessions(prev => prev.filter(s => s.session_id !== sessionId));
    } catch (err) {
      console.error(err);
    }
  };

  const handleNewSessionLocal = async (e, docId) => {
    e.stopPropagation();
    try {
      await onCreateSessionForDoc?.(docId);
    } catch (err) {
      console.error(err);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    for (const file of files) {
      await handleUpload(file);
    }
  };

  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files);
    for (const file of files) {
      await handleUpload(file);
    }
    e.target.value = '';
  };

  const handleUpload = async (file) => {
    try {
      setUploadProgress(0);
      const result = await uploadDocument(file, (progress) => {
        setUploadProgress(progress);
      });
      
      if (result.status === 'duplicate') {
        onToast?.('Document already exists', 'info');
        onRefreshDocs?.();
        return;
      }
      
      if (result.status === 'ready') {
        onToast?.(`${result.filename} processed — ${result.total_chunks} chunks`, 'success');
        onRefreshDocs?.();
        return;
      }
      
      // Status is 'processing' — poll until ready or failed
      if (result.status === 'processing' && result.doc_id) {
        onToast?.(`${result.filename} uploaded — processing...`, 'info');
        onRefreshDocs?.();
        
        // Poll every 2s for up to 5 minutes
        const maxAttempts = 150;
        let attempts = 0;
        
        const poll = async () => {
          while (attempts < maxAttempts) {
            attempts++;
            await new Promise(r => setTimeout(r, 2000));
            try {
              const doc = await getDocument(result.doc_id);
              if (doc.status === 'ready') {
                onToast?.(`${result.filename} ready — ${doc.total_chunks} chunks`, 'success');
                onRefreshDocs?.();
                return;
              } else if (doc.status.startsWith('failed')) {
                onToast?.(`Processing failed for ${result.filename}: ${doc.status.replace('failed: ', '')}`, 'error');
                onRefreshDocs?.();
                return;
              }
              // Still processing — continue polling
            } catch (err) {
              console.error('Polling error:', err);
            }
          }
          onToast?.(`Processing is taking longer than expected for ${result.filename}`, 'warning');
        };
        
        // Run polling in background — don't block the UI
        poll();
        return;
      }

      // Any other status
      onToast?.(`Upload: ${result.message}`, 'info');
      onRefreshDocs?.();
    } catch (error) {
      onToast?.('Upload failed: ' + (error.response?.data?.detail || error.message), 'error');
    } finally {
      setUploadProgress(null);
    }
  };

  const handleDelete = async (docId, filename) => {
    try {
      await deleteDocument(docId);
      onToast?.(`${filename} removed`, 'success');
      onRefreshDocs?.();
    } catch (error) {
      onToast?.('Failed to delete document', 'error');
    }
  };

  const getDocClass = (type) => {
    switch (type) {
      case 'pdf': return 'pdf';
      case 'docx': case 'doc': return 'docx';
      case 'txt': return 'txt';
      case 'pptx': case 'ppt': return 'ppt';
      default: return '';
    }
  };

  const isConnected = health?.llm_connected;

  return (
    <aside className="sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <span className="logo-icon">◆</span>
          <h1>RAG Agent</h1>
        </div>
        <div className="sidebar-subtitle">Document intelligence</div>
      </div>

      {/* Upload zone */}
      <div className="upload-section">
        <div
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <span className="upload-icon">↑</span>
          <div className="upload-text">Drop files or click to upload</div>
          <div className="upload-hint">PDF, DOCX, PPTX, TXT</div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.doc,.txt,.ppt,.pptx"
            multiple
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
        </div>

        {uploadProgress !== null && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{ width: `${uploadProgress}%` }} />
            </div>
            <div className="progress-text">{uploadProgress === 100 ? 'Processing...' : `${uploadProgress}%`}</div>
          </div>
        )}
      </div>

      {/* Documents list */}
      <div className="documents-section">
        <div className="documents-section-header">
          <h3>Documents</h3>
          <span className="doc-count">{documents.length}</span>
        </div>

        {documents.length === 0 ? (
          <div className="no-documents">No documents uploaded yet</div>
        ) : (
          documents.map((doc) => {
            const isActive = activeDocumentId === doc.doc_id;
            const isExpanded = expandedDocId === doc.doc_id;
            return (
              <div key={doc.doc_id} className="document-item-wrapper">
                <div
                  className={`document-item ${isActive ? 'active' : ''}`}
                  onClick={() => handleDocumentClickLocal(doc)}
                  title={`Click to view conversations for "${doc.filename}"`}
                >
                  <div className={`doc-icon ${getDocClass(doc.document_type)}`}>⬒</div>
                  <div className="doc-info">
                    <div className="doc-name">{doc.filename}</div>
                    <div className="doc-meta">
                      {doc.total_chunks} chunks · {doc.document_type?.toUpperCase()}
                    </div>
                  </div>
                  <div className={`doc-status ${doc.status.startsWith('failed') ? 'failed' : doc.status}`} />
                  <button
                    className="doc-delete-btn"
                    onClick={(e) => { e.stopPropagation(); handleDelete(doc.doc_id, doc.filename); }}
                    title="Remove"
                  >
                    ×
                  </button>
                </div>

                {isExpanded && (
                  <div className="document-sessions-list">
                    <button
                      className="new-session-btn"
                      onClick={(e) => handleNewSessionLocal(e, doc.doc_id)}
                    >
                      <span>＋</span> New chat session
                    </button>
                    {loadingSessions ? (
                      <div className="no-documents">Loading conversations...</div>
                    ) : docSessions.length === 0 ? (
                      <div className="no-documents">No past conversations</div>
                    ) : (
                      docSessions.map((session) => {
                        const isSessionActive = activeSessionId === session.session_id;
                        const dateStr = new Date(session.created_at).toLocaleDateString(undefined, {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit'
                        });
                        return (
                          <div
                            key={session.session_id}
                            className={`doc-session-item ${isSessionActive ? 'active' : ''}`}
                            onClick={() => onSelectSession?.(session.session_id)}
                          >
                            <div className="doc-session-item-content">
                              <div className="doc-session-item-preview" title={session.preview}>
                                {session.preview || 'Empty conversation'}
                              </div>
                              <div className="doc-session-item-date">{dateStr}</div>
                            </div>
                            <button
                              className="doc-session-item-delete"
                              onClick={(e) => handleDeleteSessionLocal(e, session.session_id)}
                              title="Delete conversation"
                            >
                              ×
                            </button>
                          </div>
                        );
                      })
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* View Switcher */}
      <div className="view-switcher" style={{ padding: '0 16px 16px 16px', display: 'flex', gap: '8px' }}>
        <button 
          className={`new-session-btn ${view === 'chat' ? 'active' : ''}`} 
          style={{ flex: 1, justifyContent: 'center', background: view === 'chat' ? 'var(--accent-subtle)' : 'transparent' }}
          onClick={() => setView('chat')}
        >
          Chat
        </button>
        <button 
          className={`new-session-btn ${view === 'usage' ? 'active' : ''}`} 
          style={{ flex: 1, justifyContent: 'center', background: view === 'usage' ? 'var(--accent-subtle)' : 'transparent' }}
          onClick={() => setView('usage')}
        >
          Usage
        </button>
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
        <span className="status-text">
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
        <span className="model-name">{health?.model || '—'}</span>
        <span className="footer-spacer" />
        <button
          className="theme-toggle"
          onClick={onToggleTheme}
          title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
        >
          {theme === 'light' ? '☽' : '☀'}
        </button>
      </div>
    </aside>
  );
}
