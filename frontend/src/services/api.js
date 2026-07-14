import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

// === Documents ===

export async function uploadDocument(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress
      ? (e) => onProgress(Math.round((e.loaded * 100) / e.total))
      : undefined,
  });
  return response.data;
}

export async function listDocuments() {
  const response = await api.get('/documents');
  return response.data;
}

export async function getDocument(docId) {
  const response = await api.get(`/documents/${docId}`);
  return response.data;
}

export async function deleteDocument(docId) {
  const response = await api.delete(`/documents/${docId}`);
  return response.data;
}

export async function summarizeDocument(docId) {
  const response = await api.post(`/documents/${docId}/summarize`);
  return response.data;
}

export async function getDocumentSessions(docId) {
  const response = await api.get(`/documents/${docId}/sessions`);
  return response.data;
}

// === Chat Sessions ===

export async function createSession(documentId = null) {
  const body = {};
  if (documentId) body.document_id = documentId;
  const response = await api.post('/chat/sessions', body);
  return response.data;
}

export async function listSessions() {
  const response = await api.get('/chat/sessions');
  return response.data;
}

export async function getSessionHistory(sessionId) {
  const response = await api.get(`/chat/sessions/${sessionId}`);
  return response.data;
}

export async function deleteSession(sessionId) {
  const response = await api.delete(`/chat/sessions/${sessionId}`);
  return response.data;
}

export async function sendMessage(sessionId, message, metadataFilters = {}) {
  const response = await api.post(`/chat/${sessionId}/message`, {
    message,
    metadata_filters: metadataFilters,
  });
  return response.data;
}

// === Streaming ===

export function streamMessage(sessionId, message, filters = {}, onToken, onCitations, onDone, onError, onTraceId, onStats) {
  const params = new URLSearchParams({ message });
  if (Object.keys(filters).length > 0) {
    params.set('filters', JSON.stringify(filters));
  }
  
  const url = `${API_BASE}/chat/${sessionId}/stream?${params}`;
  const eventSource = new EventSource(url);
  
  // Timeout: if no data in 90s, close and error
  let lastActivity = Date.now();
  const timeoutCheck = setInterval(() => {
    if (Date.now() - lastActivity > 90000) {
      clearInterval(timeoutCheck);
      eventSource.close();
      onError?.('Request timed out — no response from server.');
    }
  }, 5000);
  
  eventSource.onmessage = (event) => {
    lastActivity = Date.now();
    try {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'trace_id':
          onTraceId?.(data.trace_id);
          break;
        case 'token':
          onToken?.(data.content);
          break;
        case 'citations':
          onCitations?.(data.citations);
          break;
        case 'stats':
          onStats?.(data.stats);
          break;
        case 'error':
          // Show error content as a message token so user sees it
          onToken?.(data.content || 'An error occurred.');
          break;
        case 'done':
          clearInterval(timeoutCheck);
          onDone?.();
          eventSource.close();
          break;
        default:
          break;
      }
    } catch (e) {
      console.error('Failed to parse SSE event:', e);
    }
  };
  
  eventSource.onerror = (error) => {
    clearInterval(timeoutCheck);
    onError?.(error);
    eventSource.close();
  };
  
  return eventSource;
}

// === Search ===

export async function hybridSearch(query, topK = 10) {
  const response = await api.post('/search', { query, top_k: topK });
  return response.data;
}

export async function filteredSearch(query, filters = {}, topK = 10) {
  const response = await api.post('/search/filtered', {
    query,
    top_k: topK,
    filters,
  });
  return response.data;
}

// === Feedback & Tracing (Phase 6) ===

export async function submitFeedback({ trace_id, session_id, rating, comment = '' }) {
  const response = await api.post('/feedback', {
    trace_id,
    session_id,
    rating,
    comment,
  });
  return response.data;
}

export async function getTrace(traceId) {
  const response = await api.get(`/traces/${traceId}`);
  return response.data;
}

// === Tools (Phase 5) ===

export async function webSearch(query, maxResults = 5) {
  const response = await api.post('/tools/web-search', {
    query,
    max_results: maxResults,
  });
  return response.data;
}

export async function calculate(expression) {
  const response = await api.post('/tools/calculate', { expression });
  return response.data;
}

// === System ===

export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

export async function getModels() {
  const response = await api.get('/models');
  return response.data;
}

export async function getConfig() {
  const response = await api.get('/config');
  return response.data;
}

export async function getUsageStats() {
  const response = await api.get('/usage/stats');
  return response.data;
}

export default api;
