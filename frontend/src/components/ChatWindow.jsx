import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ChatWindow.css'

const EXAMPLES = [
  'Recent advances in LLM reasoning',
  'How do diffusion models work?',
  'What is retrieval-augmented generation?',
  'Transformers vs state space models',
]

function UserMessage({ msg }) {
  return (
    <div className="msg msg-user">
      <p className="msg-text">{msg.content}</p>
    </div>
  )
}

function AssistantMessage({ msg }) {
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(msg.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`msg msg-ai ${msg.isError ? 'msg-error' : ''}`}>
      <div className="msg-ai-header">
        <span className="msg-label">Answer</span>
        <button id={`copy-${msg.id}`} className="copy-btn" onClick={copy} title="Copy">
          {copied
            ? <svg viewBox="0 0 16 16" fill="none"><path d="M3 8l3.5 3.5L13 4" stroke="#4ade80" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
            : <svg viewBox="0 0 16 16" fill="none"><rect x="5" y="1" width="10" height="12" rx="2" stroke="currentColor" strokeWidth="1.2"/><rect x="1" y="4" width="10" height="12" rx="2" stroke="currentColor" strokeWidth="1.2"/></svg>
          }
        </button>
      </div>
      <div className="md">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
      </div>
    </div>
  )
}

function ThinkingDots() {
  return (
    <div className="msg msg-ai">
      <div className="msg-label">Researching</div>
      <div className="dots">
        <span /><span /><span />
      </div>
    </div>
  )
}

export default function ChatWindow({ messages, isLoading, bottomRef, onSubmit }) {
  const [value, setValue] = useState('')
  const isEmpty = messages.length === 0

  const submit = (e) => {
    e?.preventDefault()
    const q = value.trim()
    if (q && !isLoading) {
      onSubmit(q)
      setValue('')
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  return (
    <div className="chat">
      {/* Header */}
      <header className="chat-header">
        <div className="logo">
          <div className="logo-dot" />
          <span className="logo-name">Research Assistant</span>
        </div>
        <p className="logo-desc">Ask anything — I'll search ArXiv and your paper database to answer.</p>
      </header>

      {/* Messages or empty state */}
      <div className="messages">
        {isEmpty && !isLoading && (
          <div className="empty">
            <p className="empty-title">What do you want to know?</p>
            <p className="empty-sub">Try one of these to get started</p>
            <div className="examples">
              {EXAMPLES.map((q, i) => (
                <button
                  key={i}
                  id={`example-${i}`}
                  className="example-btn"
                  onClick={() => onSubmit(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) =>
          msg.role === 'user'
            ? <UserMessage key={msg.id} msg={msg} />
            : <AssistantMessage key={msg.id} msg={msg} />
        )}

        {isLoading && <ThinkingDots />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-wrap">
        <form className="input-form" onSubmit={submit}>
          <textarea
            id="query-input"
            className="input-ta"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a research question…"
            rows={1}
            disabled={isLoading}
          />
          <button
            id="send-btn"
            type="submit"
            className={`send-btn ${value.trim() ? 'active' : ''}`}
            disabled={isLoading || !value.trim()}
          >
            {isLoading
              ? <div className="spinner" />
              : <svg viewBox="0 0 20 20" fill="none"><path d="M4 10h12M11 5l5 5-5 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
            }
          </button>
        </form>
        <p className="input-hint">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  )
}
