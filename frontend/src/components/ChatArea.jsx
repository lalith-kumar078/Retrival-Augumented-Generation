import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { streamMessage, sendMessage, submitFeedback } from '../services/api';

/* ── Citation helpers ─────────────────────────────────────────── */

/**
 * Parse markdown text and split it into segments:
 * plain text and citation markers like [1], [2], [1][3], etc.
 */
function parseCitationMarkers(text) {
  // Match [n] patterns — possibly consecutive like [1][2]
  const parts = [];
  // Match [n] individual markers AND [n, m, ...] comma-separated markers
  const regex = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
    }
    // Split comma-separated numbers into individual cite entries
    const nums = match[1].split(/\s*,\s*/);
    for (const num of nums) {
      parts.push({ type: 'cite', value: parseInt(num, 10) });
    }
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIndex) });
  }
  return parts;
}

/**
 * Deduplicate citations by (filename, page_number).
 * Returns { deduped: [...], countMap: { 'file|page': count } }
 */
function deduplicateCitations(citations) {
  if (!citations?.length) return { deduped: [], countMap: {} };

  const seen = new Map();
  const countMap = {};

  for (const cite of citations) {
    const key = `${cite.filename}|${cite.page_number}`;
    countMap[key] = (countMap[key] || 0) + 1;
    if (!seen.has(key)) {
      seen.set(key, cite);
    }
  }

  return { deduped: Array.from(seen.values()), countMap };
}

/**
 * Inline component that renders text with [n] markers as superscript pills.
 */
function CitedText({ children, onCiteClick }) {
  if (typeof children !== 'string') return children;
  const parts = parseCitationMarkers(children);
  if (parts.length === 1 && parts[0].type === 'text') return children;

  return parts.map((part, i) => {
    if (part.type === 'cite') {
      return (
        <sup
          key={i}
          className="cite-marker"
          onClick={(e) => { e.stopPropagation(); onCiteClick?.(part.value); }}
          title={`Jump to source [${part.value}]`}
        >
          {part.value}
        </sup>
      );
    }
    return <span key={i}>{part.value}</span>;
  });
}

/**
 * Custom renderers for ReactMarkdown that inject citation pill handling.
 */
function makeMarkdownComponents(onCiteClick) {
  // Process children that may contain citation markers
  const processChildren = (children) => {
    if (!children) return children;
    return React.Children.map(children, (child) => {
      if (typeof child === 'string') {
        return <CitedText onCiteClick={onCiteClick}>{child}</CitedText>;
      }
      return child;
    });
  };

  return {
    p: ({ children }) => <p>{processChildren(children)}</p>,
    li: ({ children }) => <li>{processChildren(children)}</li>,
    strong: ({ children }) => <strong>{processChildren(children)}</strong>,
    em: ({ children }) => <em>{processChildren(children)}</em>,
  };
}


/* ── Main component ───────────────────────────────────────────── */

export default function ChatArea({ 
  sessionId, 
  messages, 
  setMessages, 
  onNewChat, 
  isStreaming, 
  setIsStreaming,
  onToast,
  documentFilename,
  documentId,
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const citationRefsMap = useRef({});

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const handleCiteClick = useCallback((sourceNum) => {
    const el = citationRefsMap.current[sourceNum];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      el.classList.add('cite-highlight');
      setTimeout(() => el.classList.remove('cite-highlight'), 1500);
    }
  }, []);

  const markdownComponents = useMemo(() => makeMarkdownComponents(handleCiteClick), [handleCiteClick]);

  const handleSend = async () => {
    if (!input.trim() || !sessionId || isStreaming) return;
    
    const userMessage = input.trim();
    setInput('');
    
    // Add user message immediately
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      citations: [],
      timestamp: new Date().toISOString(),
    }]);

    // Add placeholder for assistant
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      citations: [],
      streaming: true,
      trace_id: null,
      feedback: null,
      timestamp: new Date().toISOString(),
    }]);
    
    setIsStreaming(true);

    let hasReceivedTokens = false;

    // Use streaming
    streamMessage(
      sessionId,
      userMessage,
      {},
      // onToken
      (token) => {
        hasReceivedTokens = true;
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: last.content + token
            };
          }
          return updated;
        });
      },
      // onCitations
      (citations) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              citations: citations
            };
          }
          return updated;
        });
      },
      // onDone
      () => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last) {
            updated[updated.length - 1] = {
              ...last,
              streaming: false
            };
          }
          return updated;
        });
        setIsStreaming(false);
      },
      // onError
      (error) => {
        console.error('Stream error:', error);
        const errorMsg = typeof error === 'string' ? error : 'Connection error. Retrying...';
        if (typeof error === 'string') {
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = {
                ...last,
                content: errorMsg,
                streaming: false
              };
            }
            return updated;
          });
          setIsStreaming(false);
          onToast?.(errorMsg, 'error');
        } else {
          if (hasReceivedTokens) {
            setMessages(prev => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last && last.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...last,
                  streaming: false
                };
              }
              return updated;
            });
            setIsStreaming(false);
          } else {
            handleFallbackSend(userMessage);
          }
        }
      },
      // onTraceId
      (traceId) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              trace_id: traceId
            };
          }
          return updated;
        });
      },
      // onStats
      (stats) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              stats: stats
            };
          }
          return updated;
        });
      }
    );
  };

  const handleFallbackSend = async (userMessage) => {
    try {
      const response = await sendMessage(sessionId, userMessage);
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: response.message.content,
            citations: response.message.citations || [],
            trace_id: response.trace_id || null,
            streaming: false
          };
        }
        return updated;
      });
    } catch (error) {
      onToast?.('Failed to get response', 'error');
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: 'Sorry, I encountered an error. Please try again.',
            streaming: false
          };
        }
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleFeedback = async (msgIndex, rating) => {
    const msg = messages[msgIndex];
    if (!msg || !msg.trace_id || msg.feedback === rating) return;

    try {
      await submitFeedback({
        trace_id: msg.trace_id,
        session_id: sessionId,
        rating,
      });

      setMessages(prev => {
        const updated = [...prev];
        updated[msgIndex] = { ...updated[msgIndex], feedback: rating };
        return updated;
      });

      onToast?.(`Feedback recorded: ${rating === 'up' ? '👍' : '👎'}`, 'success');
    } catch (error) {
      onToast?.('Failed to submit feedback', 'error');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const hasMessages = messages.length > 0;

  return (
    <main className="main-content">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header-left">
          <h2>Chat</h2>
          {sessionId && (
            <span className="chat-session-id">
              {sessionId.slice(0, 8)}
            </span>
          )}
          {documentFilename && (
            <span className="chat-doc-scope" title={`Scoped to document: ${documentFilename}`}>
              <span className="scope-icon">⬒</span>
              {documentFilename.length > 30
                ? documentFilename.slice(0, 27) + '…'
                : documentFilename}
            </span>
          )}
          {!documentId && sessionId && (
            <span className="chat-scope-all" title="Searching across all documents">
              All documents
            </span>
          )}
        </div>
        <button className="new-chat-btn" onClick={onNewChat}>
          + New Chat
        </button>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {!hasMessages ? (
          <div className="empty-state">
            <h3>
              {documentFilename
                ? `Ask about "${documentFilename}"`
                : 'Ask your documents anything'}
            </h3>
            <p>
              {documentFilename
                ? 'This chat is scoped to the selected document. Answers will come from its content only.'
                : 'Upload files in the sidebar, then ask questions. Click a document to chat about it specifically.'}
            </p>
            <div className="empty-features">
              <div className="feature-card clickable" onClick={() => setInput('Summarize the key points from my documents')}>
                <span className="feature-icon">⊕</span>
                <h4>Hybrid Search</h4>
                <p>Semantic + keyword search with rank fusion</p>
              </div>
              <div className="feature-card clickable" onClick={() => setInput('What are the main findings? Cite your sources.')}>
                <span className="feature-icon">⎘</span>
                <h4>Source Citations</h4>
                <p>Every claim linked to filename & page</p>
              </div>
              <div className="feature-card clickable" onClick={() => setInput('Give me a detailed overview of the uploaded content')}>
                <span className="feature-icon">↯</span>
                <h4>Streaming</h4>
                <p>Token-by-token responses in real time</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg, index) => {
              const { deduped: dedupedCitations, countMap } = deduplicateCitations(msg.citations);
              
              return (
                <div key={index} className={`message ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? '→' : '◇'}
                  </div>
                  <div className="message-body">
                    <div className="message-sender">
                      {msg.role === 'user' ? 'You' : 'Assistant'}
                    </div>
                    <div className="message-content">
                      {msg.role === 'assistant' && msg.content ? (
                        <>
                          <ReactMarkdown components={markdownComponents}>
                            {msg.content}
                          </ReactMarkdown>
                          {msg.streaming && <span className="streaming-cursor" />}
                        </>
                      ) : msg.role === 'assistant' && msg.streaming ? (
                        <div className="typing-indicator">
                          <span /><span /><span />
                        </div>
                      ) : (
                        <p>{msg.content}</p>
                      )}
                    </div>
                    
                    {/* Deduplicated citations */}
                    {dedupedCitations.length > 0 && !msg.streaming && (
                      <div className="citations">
                        {dedupedCitations.map((cite, i) => {
                          const key = `${cite.filename}|${cite.page_number}`;
                          const count = countMap[key] || 1;
                          return (
                            <span
                              key={i}
                              className="citation-chip"
                              title={cite.snippet}
                              ref={(el) => {
                                if (el) citationRefsMap.current[i + 1] = el;
                              }}
                            >
                              <span className="citation-number">{i + 1}</span>
                              <span className="citation-label">
                                {cite.filename}, p.{cite.page_number}
                                {cite.line_number ? `, line ${cite.line_number}` : ''}
                              </span>
                              {count > 1 && (
                                <span className="citation-count">{count}×</span>
                              )}
                            </span>
                          );
                        })}
                      </div>
                    )}
                    
                    {/* Stats footer (DeepSeek style) */}
                    {msg.role === 'assistant' && msg.stats && !msg.streaming && (
                      <details className="message-stats-footer">
                        <summary>
                          Answered in {(msg.stats.duration_ms / 1000).toFixed(1)}s · {msg.stats.usage?.total_tokens?.toLocaleString() || 0} tokens · Retrieved {msg.stats.chunks_retrieved} chunks
                        </summary>
                        <div className="stats-breakdown">
                          <div className="stat-item"><span>Retrieval:</span> <span>{msg.stats.retrieval_ms.toFixed(0)}ms</span></div>
                          <div className="stat-item"><span>Reranking:</span> <span>{msg.stats.rerank_ms.toFixed(0)}ms</span></div>
                          <div className="stat-item"><span>Generation:</span> <span>{msg.stats.generation_ms.toFixed(0)}ms</span></div>
                          <div className="stat-item"><span>Prompt Tokens:</span> <span>{msg.stats.usage?.prompt_tokens?.toLocaleString() || 0}</span></div>
                          <div className="stat-item"><span>Completion Tokens:</span> <span>{msg.stats.usage?.completion_tokens?.toLocaleString() || 0}</span></div>
                        </div>
                      </details>
                    )}

                    {/* Feedback buttons */}
                    {msg.role === 'assistant' && !msg.streaming && msg.content && (
                      <div className="message-actions">
                        <button
                          className={`feedback-btn ${msg.feedback === 'up' ? 'active' : ''}`}
                          onClick={() => handleFeedback(index, 'up')}
                          title="Helpful"
                        >
                          ↑
                        </button>
                        <button
                          className={`feedback-btn ${msg.feedback === 'down' ? 'active' : ''}`}
                          onClick={() => handleFeedback(index, 'down')}
                          title="Not helpful"
                        >
                          ↓
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="input-area">
        <div className="input-wrapper">
          <div className="input-container">
            <textarea
              ref={textareaRef}
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                !sessionId 
                  ? 'Initializing session...' 
                  : documentFilename
                    ? `Ask about ${documentFilename}…`
                    : 'Ask about your documents…'
              }
              disabled={!sessionId || isStreaming}
              rows={1}
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!input.trim() || !sessionId || isStreaming}
              title="Send message"
            >
              {isStreaming ? '…' : '→'}
            </button>
          </div>
          <div className="input-hint">
            {documentFilename
              ? `Answers grounded in: ${documentFilename}`
              : 'Answers are grounded in your uploaded documents.'}
          </div>
        </div>
      </div>
    </main>
  );
}
