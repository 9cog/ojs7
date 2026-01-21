import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar.jsx'
import { 
  MessageCircle, 
  X, 
  Send, 
  Bot, 
  User, 
  Minimize2, 
  Maximize2,
  RefreshCw,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react'

const CHATBOT_API_BASE = 'https://p9hwiqcl9n19.manus.space/api/chatbot/v1'

const ChatbotWidget = ({ userContext = {} }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [agentType, setAgentType] = useState('support')
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  const agentInfo = {
    submission: {
      name: 'Submission Assistant',
      avatar: '📝',
      color: 'bg-blue-500',
      description: 'Helps with manuscript submissions'
    },
    editorial: {
      name: 'Editorial Support',
      avatar: '✏️',
      color: 'bg-green-500',
      description: 'Assists editors with workflow'
    },
    review: {
      name: 'Review Facilitator',
      avatar: '🔍',
      color: 'bg-purple-500',
      description: 'Guides peer review process'
    },
    support: {
      name: 'Technical Support',
      avatar: '🛠️',
      color: 'bg-orange-500',
      description: 'Provides system help'
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const createSession = async () => {
    try {
      setIsLoading(true)
      setError(null)
      
      const response = await fetch(`${CHATBOT_API_BASE}/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: userContext.userId || 1,
          context: {
            page: userContext.page || 'dashboard',
            userRole: userContext.userRole || 'author',
            ...userContext
          }
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.success) {
        setSessionId(data.data.sessionId)
        setAgentType(data.data.agentType)
        setIsConnected(true)
        
        // Add welcome message
        setMessages([{
          id: 'welcome',
          role: 'assistant',
          content: data.data.welcomeMessage,
          timestamp: new Date().toISOString(),
          agentType: data.data.agentType
        }])
      } else {
        throw new Error(data.error?.message || 'Failed to create session')
      }
    } catch (err) {
      console.error('Failed to create chatbot session:', err)
      setError('Failed to connect to chatbot service. Please try again.')
      setIsConnected(false)
    } finally {
      setIsLoading(false)
    }
  }

  const sendMessage = async (message) => {
    if (!sessionId || !message.trim()) return

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message.trim(),
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      const response = await fetch(`${CHATBOT_API_BASE}/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message.trim(),
          messageType: 'text'
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.success) {
        const assistantMessage = {
          id: data.data.messageId,
          role: 'assistant',
          content: data.data.response.text,
          timestamp: new Date().toISOString(),
          agentType: data.data.agentType,
          confidence: data.data.confidence,
          actions: data.data.response.actions || [],
          suggestions: data.data.response.suggestions || []
        }
        
        setMessages(prev => [...prev, assistantMessage])
      } else {
        throw new Error(data.error?.message || 'Failed to send message')
      }
    } catch (err) {
      console.error('Failed to send message:', err)
      const errorMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
        isError: true
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSuggestionClick = (suggestion) => {
    sendMessage(suggestion)
  }

  const handleActionClick = (action) => {
    if (action.action === 'navigate') {
      // In a real app, this would use React Router
      console.log('Navigate to:', action.data.url)
    } else if (action.action === 'open_modal') {
      console.log('Open modal:', action.data.modal)
    }
  }

  const resetChat = () => {
    setMessages([])
    setSessionId(null)
    setIsConnected(false)
    setError(null)
    if (isOpen) {
      createSession()
    }
  }

  const handleOpen = () => {
    setIsOpen(true)
    if (!sessionId && !isLoading) {
      createSession()
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(inputMessage)
    }
  }

  const currentAgent = agentInfo[agentType] || agentInfo.support

  if (!isOpen) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        <Button 
          size="lg" 
          className="rounded-full h-14 w-14 shadow-lg bg-primary hover:bg-primary/90 transition-all duration-200 hover:scale-105"
          onClick={handleOpen}
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      </div>
    )
  }

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <Card className={`w-96 shadow-xl transition-all duration-300 ${isMinimized ? 'h-16' : 'h-[500px]'}`}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 bg-primary text-primary-foreground rounded-t-lg">
          <div className="flex items-center space-x-2">
            <div className={`w-8 h-8 rounded-full ${currentAgent.color} flex items-center justify-center text-white text-sm`}>
              {currentAgent.avatar}
            </div>
            <div>
              <CardTitle className="text-sm">{currentAgent.name}</CardTitle>
              <p className="text-xs opacity-90">{currentAgent.description}</p>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            {isConnected && (
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 text-primary-foreground hover:bg-primary-foreground/20"
              onClick={() => setIsMinimized(!isMinimized)}
            >
              {isMinimized ? <Maximize2 className="h-3 w-3" /> : <Minimize2 className="h-3 w-3" />}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 text-primary-foreground hover:bg-primary-foreground/20"
              onClick={resetChat}
            >
              <RefreshCw className="h-3 w-3" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 text-primary-foreground hover:bg-primary-foreground/20"
              onClick={() => setIsOpen(false)}
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        </CardHeader>

        {!isMinimized && (
          <CardContent className="flex flex-col h-[calc(500px-80px)] p-0">
            {error && (
              <div className="p-4 bg-red-50 border-b border-red-200">
                <p className="text-sm text-red-600">{error}</p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2"
                  onClick={createSession}
                >
                  Retry Connection
                </Button>
              </div>
            )}

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`flex items-start space-x-2 max-w-[80%] ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    <Avatar className="w-6 h-6">
                      {message.role === 'user' ? (
                        <AvatarFallback className="bg-secondary text-secondary-foreground">
                          <User className="h-3 w-3" />
                        </AvatarFallback>
                      ) : (
                        <AvatarFallback className={`${currentAgent.color} text-white`}>
                          <Bot className="h-3 w-3" />
                        </AvatarFallback>
                      )}
                    </Avatar>
                    <div className={`rounded-lg p-3 ${
                      message.role === 'user' 
                        ? 'bg-primary text-primary-foreground' 
                        : message.isError 
                          ? 'bg-red-50 text-red-700 border border-red-200'
                          : 'bg-muted'
                    }`}>
                      <p className="text-sm">{message.content}</p>
                      {message.confidence && (
                        <div className="mt-2 flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            Confidence: {Math.round(message.confidence * 100)}%
                          </Badge>
                        </div>
                      )}
                      {message.actions && message.actions.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {message.actions.map((action, index) => (
                            <Button
                              key={index}
                              variant="outline"
                              size="sm"
                              className="text-xs"
                              onClick={() => handleActionClick(action)}
                            >
                              {action.label}
                            </Button>
                          ))}
                        </div>
                      )}
                      {message.suggestions && message.suggestions.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {message.suggestions.slice(0, 3).map((suggestion, index) => (
                            <Button
                              key={index}
                              variant="ghost"
                              size="sm"
                              className="text-xs h-6 px-2 bg-background/50 hover:bg-background"
                              onClick={() => handleSuggestionClick(suggestion)}
                            >
                              {suggestion}
                            </Button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-center space-x-2">
                    <Avatar className="w-6 h-6">
                      <AvatarFallback className={`${currentAgent.color} text-white`}>
                        <Bot className="h-3 w-3" />
                      </AvatarFallback>
                    </Avatar>
                    <div className="bg-muted rounded-lg p-3">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="border-t p-4">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message..."
                  className="flex-1 px-3 py-2 border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                  disabled={isLoading || !isConnected}
                />
                <Button
                  size="sm"
                  onClick={() => sendMessage(inputMessage)}
                  disabled={isLoading || !inputMessage.trim() || !isConnected}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  )
}

export default ChatbotWidget

