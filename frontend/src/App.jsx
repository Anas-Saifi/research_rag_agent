import { useState, useRef, useEffect } from 'react'
import ChatWindow from './components/ChatWindow'
import './App.css'

export default function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [status, setStatus] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const sendQuery = async (query) => {
    if (!query.trim() || isLoading) return

    const userMsg = { id: Date.now() + '-u', role: 'user', content: query, ts: new Date() }
    setMessages((prev) => [...prev, userMsg])
    setIsLoading(true)
    setStatus('thinking')

    try {
      const res = await fetch(
      'https://backend-411148586126.asia-south1.run.app/query',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      }
    )
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Server error')
      }
      const data = await res.json()
      setMessages((prev) => [...prev, {
        id: Date.now() + '-a',
        role: 'assistant',
        content: data.response,
        ts: new Date(),
      }])
    } catch (err) {
      setMessages((prev) => [...prev, {
        id: Date.now() + '-e',
        role: 'assistant',
        content: `**Error:** ${err.message}\n\nMake sure the backend is running:\n\`\`\`\nuv run uvicorn api:app --reload\n\`\`\``,
        ts: new Date(),
        isError: true,
      }])
    } finally {
      setIsLoading(false)
      setStatus(null)
    }
  }

  return (
    <div className="app">
      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        bottomRef={bottomRef}
        onSubmit={sendQuery}
      />
    </div>
  )
}
