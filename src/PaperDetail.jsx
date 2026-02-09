import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import papers10000 from './papers.json'
import papers1000 from './papers1000.json'
import './PaperDetail.css'

// Combine all papers for lookup
const allPapers = [...papers10000, ...papers1000]

async function makeIntuitive(title, abstract, calibrationData, misunderstandingSummaries = [], onChunk) {
    const knowledgeContext = calibrationData
        .map(({ term, rating }) => `- "${term}": ${rating}/10 familiarity`)
        .join('\n')

    const lowFamiliarity = calibrationData
        .filter(({ rating }) => rating <= 4)
        .map(({ term }) => term)

    const highFamiliarity = calibrationData
        .filter(({ rating }) => rating >= 7)
        .map(({ term }) => term)

    const misunderstandingContext = misunderstandingSummaries.length > 0
        ? `\n\nPREVIOUS MISUNDERSTANDINGS (from this session's clarification chats):\n${misunderstandingSummaries.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\nUse these insights to provide even clearer explanations.`
        : ''

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`
        },
        body: JSON.stringify({
            model: 'gpt-4o',
            messages: [{
                role: 'user',
                content: `You are helping someone understand a research paper. Here is their self-reported familiarity with various ML concepts:

${knowledgeContext}

CONCEPTS THEY LIKELY DON'T KNOW (rated ≤4): ${lowFamiliarity.length > 0 ? lowFamiliarity.join(', ') : 'None identified'}
CONCEPTS THEY KNOW WELL (rated ≥7): ${highFamiliarity.length > 0 ? highFamiliarity.join(', ') : 'None identified'}${misunderstandingContext}

Your task: Rewrite this paper's title and abstract for this reader.

IMPORTANT GUIDELINES:
1. When the paper uses a concept the reader likely doesn't know, EXPLICITLY explain what it means in parentheses or with a brief clarification.
2. For concepts the reader knows well, you can use them directly without explanation.
3. Be specific - don't say things like "learning differences instead of new functions" without explaining what "functions" means in this context if they wouldn't know.
4. Use concrete analogies for unfamiliar concepts.
5. Don't over-explain concepts they already understand well.

TITLE: ${title}

ABSTRACT: ${abstract || 'No abstract available.'}

Respond in this exact format:
SIMPLE TITLE: [your simplified title]
SIMPLE ABSTRACT: [your simplified abstract, 3-4 sentences max, with explicit explanations where needed]`
            }],
            max_tokens: 500,
            stream: true
        })
    })

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let fullContent = ''

    while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n').filter(line => line.trim() !== '')

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6)
                if (data === '[DONE]') continue

                try {
                    const parsed = JSON.parse(data)
                    const content = parsed.choices?.[0]?.delta?.content
                    if (content) {
                        fullContent += content
                        if (onChunk) onChunk(fullContent)
                    }
                } catch (e) {
                    // Skip malformed chunks
                }
            }
        }
    }

    // Parse the final result
    const titleMatch = fullContent.match(/SIMPLE TITLE:\s*(.+?)(?=\n|SIMPLE ABSTRACT)/is)
    const abstractMatch = fullContent.match(/SIMPLE ABSTRACT:\s*(.+)/is)

    return {
        title: titleMatch ? titleMatch[1].trim() : title,
        abstract: abstractMatch ? abstractMatch[1].trim() : abstract
    }
}

async function chatAboutAbstract(paper, simplified, chatHistory, userMessage, onChunk) {
    const messages = [
        {
            role: 'system',
            content: `You are a helpful research paper explainer. The user is reading this paper:

TITLE: ${paper.title}
ABSTRACT: ${paper.abstract || 'No abstract available.'}

You previously simplified it as:
SIMPLIFIED TITLE: ${simplified.title}
SIMPLIFIED ABSTRACT: ${simplified.abstract}

Answer the user's clarification questions concisely. Be helpful and educational.`
        },
        ...chatHistory.map(msg => ({
            role: msg.role,
            content: msg.content
        })),
        { role: 'user', content: userMessage }
    ]

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`
        },
        body: JSON.stringify({
            model: 'gpt-4o',
            messages,
            max_tokens: 400,
            stream: true
        })
    })

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let fullContent = ''

    while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n').filter(line => line.trim() !== '')

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6)
                if (data === '[DONE]') continue

                try {
                    const parsed = JSON.parse(data)
                    const content = parsed.choices?.[0]?.delta?.content
                    if (content) {
                        fullContent += content
                        onChunk(fullContent)
                    }
                } catch (e) {
                    // Skip malformed chunks
                }
            }
        }
    }

    return fullContent
}

function PaperDetail() {
    const { paperId } = useParams()
    const navigate = useNavigate()

    const [paper, setPaper] = useState(null)
    const [calibrationData, setCalibrationData] = useState(null)
    const [simplified, setSimplified] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [streamingSimplified, setStreamingSimplified] = useState('')

    // Chat state
    const [chatHistory, setChatHistory] = useState([])
    const [chatInput, setChatInput] = useState('')
    const [chatLoading, setChatLoading] = useState(false)
    const [streamingContent, setStreamingContent] = useState('')
    const chatEndRef = useRef(null)

    // Find paper and load calibration data
    useEffect(() => {
        const foundPaper = allPapers.find(p => p.paperId === paperId)
        if (!foundPaper) {
            setError('Paper not found')
            setLoading(false)
            return
        }
        setPaper(foundPaper)

        const saved = localStorage.getItem('calibrationData')
        if (!saved) {
            // Redirect to app to calibrate first
            navigate('/app')
            return
        }
        setCalibrationData(JSON.parse(saved))
    }, [paperId, navigate])

    // Generate simplified version automatically with streaming
    useEffect(() => {
        if (!paper || !calibrationData) return

        setStreamingSimplified('')
        makeIntuitive(
            paper.title,
            paper.abstract,
            calibrationData,
            [],
            (partialContent) => setStreamingSimplified(partialContent)
        )
            .then(result => {
                setSimplified(result)
                setStreamingSimplified('')
                setLoading(false)
            })
            .catch(err => {
                setError(err.message)
                setLoading(false)
            })
    }, [paper, calibrationData])

    // Auto-scroll chat
    useEffect(() => {
        if (chatEndRef.current) {
            chatEndRef.current.scrollIntoView({ behavior: 'smooth' })
        }
    }, [chatHistory, streamingContent])

    const handleSendMessage = async () => {
        if (!chatInput.trim() || chatLoading || !simplified) return

        const userMessage = chatInput.trim()
        setChatInput('')
        setChatHistory(prev => [...prev, { role: 'user', content: userMessage }])
        setChatLoading(true)
        setStreamingContent('')

        try {
            const finalContent = await chatAboutAbstract(
                paper,
                simplified,
                chatHistory,
                userMessage,
                (partialContent) => setStreamingContent(partialContent)
            )
            setChatHistory(prev => [...prev, { role: 'assistant', content: finalContent }])
            setStreamingContent('')
        } catch (err) {
            setChatHistory(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }])
            setStreamingContent('')
        }

        setChatLoading(false)
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSendMessage()
        }
    }

    if (error) {
        return (
            <div className="paper-detail-page">
                <div className="error-container">
                    <h1>Error</h1>
                    <p>{error}</p>
                    <Link to="/app" className="back-link">← Back to papers</Link>
                </div>
            </div>
        )
    }

    if (!paper) {
        return (
            <div className="paper-detail-page">
                <div className="loading-container">
                    <p>Loading paper...</p>
                </div>
            </div>
        )
    }

    return (
        <div className="paper-detail-page">
            <header className="detail-header">
                <Link to="/app" className="back-link">← Back to papers</Link>
                <div className="paper-meta">
                    <span className="year-badge">{paper.year}</span>
                    <span className="venue-badge">{paper.venue}</span>
                    <span className="citations-badge">{paper.citationCount.toLocaleString()} citations</span>
                </div>
            </header>

            <div className="split-container">
                {/* Left side - Original */}
                <div className="split-pane original-pane">
                    <div className="pane-header">
                        <span className="pane-label">Original</span>
                    </div>
                    <div className="pane-content">
                        <h1 className="paper-title">{paper.title}</h1>
                        <p className="paper-authors">
                            {paper.authors.map(a => a.name).join(', ')}
                        </p>
                        <div className="paper-abstract">
                            <h3>Abstract</h3>
                            <p>{paper.abstract || 'No abstract available.'}</p>
                        </div>
                        <a
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="view-paper-btn"
                        >
                            View Full Paper →
                        </a>
                    </div>
                </div>

                {/* Right side - Simplified */}
                <div className="split-pane simplified-pane">
                    <div className="pane-header">
                        <span className="pane-label">Simplified for You</span>
                        {loading && <span className="generating-badge">Generating...</span>}
                    </div>
                    <div className="pane-content">
                        {simplified ? (
                            <>
                                <h1 className="paper-title simplified-title">{simplified.title}</h1>
                                <div className="paper-abstract">
                                    <h3>Simplified Abstract</h3>
                                    <p>{simplified.abstract}</p>
                                </div>

                                {/* Chat Section */}
                                <div className="chat-section">
                                    <h3>Ask Questions</h3>
                                    {(chatHistory.length > 0 || chatLoading) && (
                                        <div className="chat-messages">
                                            {chatHistory.map((msg, idx) => (
                                                <div key={idx} className={`chat-message ${msg.role}`}>
                                                    <div className="message-content">{msg.content}</div>
                                                </div>
                                            ))}
                                            {chatLoading && (
                                                <div className="chat-message assistant">
                                                    <div className={`message-content ${streamingContent ? 'streaming' : 'typing'}`}>
                                                        {streamingContent || 'Thinking...'}
                                                    </div>
                                                </div>
                                            )}
                                            <div ref={chatEndRef} />
                                        </div>
                                    )}
                                    <div className="chat-input-area">
                                        <input
                                            type="text"
                                            value={chatInput}
                                            onChange={(e) => setChatInput(e.target.value)}
                                            onKeyPress={handleKeyPress}
                                            placeholder="Type your question..."
                                            disabled={chatLoading}
                                        />
                                        <button
                                            onClick={handleSendMessage}
                                            disabled={chatLoading || !chatInput.trim()}
                                        >
                                            Send
                                        </button>
                                    </div>
                                </div>
                            </>
                        ) : loading ? (
                            <>
                                <h1 className="paper-title simplified-title">
                                    {(streamingSimplified?.match(/SIMPLE TITLE:\s*(.+?)(?=\n|SIMPLE ABSTRACT|$)/is)?.[1]?.trim()) || '\u00A0'}
                                </h1>
                                <div className="paper-abstract">
                                    <h3>Simplified Abstract</h3>
                                    <p>{streamingSimplified ? (streamingSimplified.match(/SIMPLE ABSTRACT:\s*(.+)/is)?.[1]?.trim() || '') : ''}</p>
                                </div>
                            </>
                        ) : null}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default PaperDetail
