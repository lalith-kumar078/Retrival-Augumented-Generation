import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import UsageDashboard from './components/UsageDashboard';
import Toast from './components/Toast';
import { 
  createSession, 
  listDocuments, 
  healthCheck,
  getSessionHistory,
  getDocumentSessions,
  deleteSession,
} from './services/api';

let toastId = 0;

function getInitialTheme() {
  const saved = localStorage.getItem('rag-theme');
  if (saved) return saved;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  // Sessions: { sessionId: { messages, documentId, documentFilename } }
  const [sessions, setSessions] = useState({});
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [activeDocumentId, setActiveDocumentId] = useState(null);
  const [view, setView] = useState('chat'); // 'chat' or 'usage'

  const [documents, setDocuments] = useState([]);
  const [health, setHealth] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [theme, setTheme] = useState(getInitialTheme);

  // Derived state
  const activeSession = sessions[activeSessionId] || null;
  const messages = activeSession?.messages || [];

  // Apply theme to document root
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('rag-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  }, []);

  // Toast helper
  const addToast = useCallback((message, type = 'info') => {
    const id = ++toastId;
    setToasts(prev => [...prev, { id, message, type }]);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // Initialize on mount
  useEffect(() => {
    initSession();
    fetchDocuments();
    checkHealth();
    
    // Health poll every 30s
    const healthInterval = setInterval(checkHealth, 30000);
    return () => clearInterval(healthInterval);
  }, []);

  const initSession = async (documentId = null) => {
    try {
      const session = await createSession(documentId);
      const sid = session.session_id;
      setSessions(prev => ({
        ...prev,
        [sid]: {
          messages: [],
          documentId: session.document_id || null,
          documentFilename: session.document_filename || null,
          documentIds: session.document_ids || [],
        },
      }));
      setActiveSessionId(sid);
      if (documentId) {
        setActiveDocumentId(documentId);
      } else {
        setActiveDocumentId(null);
      }
      return sid;
    } catch (error) {
      console.error('Failed to create session:', error);
      addToast('Failed to connect to backend. Is the server running?', 'error');
      return null;
    }
  };

  const fetchDocuments = async () => {
    try {
      const data = await listDocuments();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  const checkHealth = async () => {
    try {
      const data = await healthCheck();
      setHealth(data);
    } catch (error) {
      setHealth({ status: 'disconnected', llm_connected: false, vectorstore_connected: false });
    }
  };

  const handleNewChat = async () => {
    if (isStreaming) return;
    setActiveDocumentId(null);
    await initSession();
    addToast('New chat session started', 'info');
  };

  const handleDocumentClick = (doc) => {
    setActiveDocumentId(doc.doc_id);
  };

  const handleSelectSession = async (sessionId) => {
    try {
      const history = await getSessionHistory(sessionId);
      const doc = documents.find(d => d.doc_id === history.document_id);
      setSessions(prev => ({
        ...prev,
        [sessionId]: {
          messages: (history.messages || []).map(m => ({
            role: m.role,
            content: m.content,
            citations: m.citations || [],
            trace_id: m.trace_id || null,
            feedback: null,
            streaming: false,
            timestamp: m.timestamp,
          })),
          documentId: history.document_id || null,
          documentFilename: history.document_filename || doc?.filename || null,
          documentIds: history.document_id ? [history.document_id] : [],
        },
      }));
      setActiveSessionId(sessionId);
      setActiveDocumentId(history.document_id || null);
    } catch (error) {
      console.error('Failed to load session history:', error);
      addToast('Failed to load conversation history', 'error');
    }
  };

  const handleCreateSessionForDoc = async (docId) => {
    if (isStreaming) return;
    const sid = await initSession(docId);
    if (sid) {
      const doc = documents.find(d => d.doc_id === docId);
      addToast(`New chat for "${doc?.filename || 'document'}" started`, 'info');
    }
  };

  const handleDeleteSession = async (sessionId) => {
    try {
      await deleteSession(sessionId);
      setSessions(prev => {
        const updated = { ...prev };
        delete updated[sessionId];
        return updated;
      });
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setActiveDocumentId(null);
        await initSession();
      }
      addToast('Conversation deleted', 'success');
    } catch (err) {
      console.error('Failed to delete session:', err);
      addToast('Failed to delete session', 'error');
      throw err;
    }
  };

  const setMessages = useCallback((updater) => {
    setSessions(prev => {
      const sid = Object.keys(prev).find(k => k === activeSessionId);
      if (!sid || !prev[sid]) return prev;
      const newMessages = typeof updater === 'function' 
        ? updater(prev[sid].messages) 
        : updater;
      return {
        ...prev,
        [sid]: { ...prev[sid], messages: newMessages },
      };
    });
  }, [activeSessionId]);

  return (
    <div className="app-container">
      <Sidebar
        documents={documents}
        onRefreshDocs={fetchDocuments}
        health={health}
        onToast={addToast}
        theme={theme}
        onToggleTheme={toggleTheme}
        onDocumentClick={handleDocumentClick}
        activeDocumentId={activeDocumentId}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onCreateSessionForDoc={handleCreateSessionForDoc}
        onDeleteSession={handleDeleteSession}
        isStreaming={isStreaming}
        view={view}
        setView={setView}
      />
      {view === 'chat' ? (
        <ChatArea
          sessionId={activeSessionId}
          messages={messages}
          setMessages={setMessages}
          onNewChat={handleNewChat}
          isStreaming={isStreaming}
          setIsStreaming={setIsStreaming}
          onToast={addToast}
          documentFilename={activeSession?.documentFilename || null}
          documentId={activeSession?.documentId || null}
        />
      ) : (
        <UsageDashboard />
      )}
      <Toast toasts={toasts} removeToast={removeToast} />
    </div>
  );
}
