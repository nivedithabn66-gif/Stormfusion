import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, 
  X, 
  Send, 
  MessageSquare, 
  Sparkles, 
  Minimize2, 
  Maximize2,
  ChevronRight,
  Info,
  ShieldCheck,
  RotateCcw
} from 'lucide-react';
import { ChatMessage } from '../types/cyclone';
import { cycloneApi } from '../services/api';

interface CycloneAIChatbotProps {
  isFloating?: boolean;
}

export const CycloneAIChatbot: React.FC<CycloneAIChatbotProps> = ({ isFloating = true }) => {
  const [isOpen, setIsOpen] = useState(!isFloating);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      content: 'StormFusion Intelligence Assistant online. I am grounded in multi-source satellite telemetry from INSAT-3D, numerical track predictions from ConvLSTM, and coastal GIS risk models. How may I assist your operations?',
      timestamp: '14:32 IST',
      dataPoints: [
        { label: 'Current Eye', value: '14.52° N, 87.21° E' },
        { label: 'Current Wind', value: '82 kt (VSCS)' },
        { label: 'Landfall Window', value: '+72h (North AP Coast)' }
      ]
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const suggestedQuestions = [
    'Where is the cyclone now?',
    'What is the predicted track?',
    'Which districts are at risk?',
    'What is the current intensity?',
    'When is landfall expected?',
    'What safety measures should I take?'
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  const handleSendMessage = async (textToSend?: string) => {
    const query = textToSend || inputMessage;
    if (!query.trim()) return;

    const userMsg: ChatMessage = {
      id: `USR-${Date.now()}`,
      sender: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    if (!textToSend) setInputMessage('');
    setIsTyping(true);

    try {
      const response = await cycloneApi.sendChatMessage(query, messages);
      setMessages(prev => [...prev, response]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: `ERR-${Date.now()}`,
          sender: 'system',
          content: 'Operational warning: Failed to connect to AI reasoning endpoint. Retrying fallback rule engine.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  if (isFloating && !isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 p-4 rounded-2xl bg-gradient-to-br from-cyan-600 to-blue-700 text-white shadow-[0_4px_25px_rgba(2,132,199,0.35)] hover:scale-105 transition-all flex items-center gap-3 border border-cyan-400/50 group"
      >
        <span className="relative flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
          <span className="relative inline-flex rounded-full h-3 w-3 bg-white"></span>
        </span>
        <div className="flex items-center gap-2 font-mono text-sm font-black tracking-wider">
          <Bot className="w-5 h-5 group-hover:rotate-12 transition-transform" />
          <span>STORMFUSION ASSISTANT</span>
        </div>
      </button>
    );
  }

  return (
    <div
      className={`${
        isFloating
          ? 'fixed bottom-6 right-6 z-50 w-96 sm:w-[420px] max-h-[580px] shadow-2xl border-cyan-500/40'
          : 'w-full h-full max-h-[750px] border-slate-300 dark:border-slate-800'
      } command-card rounded-2xl border flex flex-col overflow-hidden bg-white dark:bg-[#0A0E1A]/95 backdrop-blur-xl shadow-2xl`}
    >
      {/* Header */}
      <div className="p-3.5 bg-slate-100 dark:bg-slate-900/90 border-b border-slate-300 dark:border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-100 dark:bg-cyan-500/20 border border-cyan-300 dark:border-cyan-500/40 flex items-center justify-center text-cyan-800 dark:text-cyan-400">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h4 className="text-xs font-black font-mono text-slate-950 dark:text-white tracking-wide">
                STORMFUSION ASSISTANT
              </h4>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 dark:bg-emerald-400 animate-pulse"></span>
            </div>
            <p className="text-[10px] font-mono text-slate-600 dark:text-slate-400 font-semibold">
              Grounded in System Telemetry & GIS
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => {
              setMessages([
                {
                  id: 'welcome',
                  sender: 'assistant',
                  content: 'Session reset. StormFusion Intelligence Assistant operational. How may I assist?',
                  timestamp: '14:32 IST'
                }
              ]);
            }}
            className="p-1 rounded text-slate-600 hover:text-slate-950 dark:text-slate-400 dark:hover:text-white transition-colors"
            title="Reset Conversation"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          {isFloating && (
            <button
              onClick={() => setIsOpen(false)}
              className="p-1 rounded text-slate-600 hover:text-slate-950 dark:text-slate-400 dark:hover:text-white transition-colors"
              title="Minimize Assistant"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Suggested Quick Questions */}
      <div className="px-3 py-2 bg-slate-100/60 dark:bg-slate-950/60 border-b border-slate-200 dark:border-slate-800 flex items-center gap-1.5 overflow-x-auto no-scrollbar">
        {suggestedQuestions.map((q, idx) => (
          <button
            key={idx}
            onClick={() => handleSendMessage(q)}
            className="whitespace-nowrap px-2.5 py-1 rounded-full bg-white dark:bg-slate-900 hover:bg-cyan-50 dark:hover:bg-slate-800 border border-slate-300 dark:border-slate-800 text-[10px] font-mono text-cyan-900 dark:text-cyan-300 hover:border-cyan-500 transition-colors shadow-xs font-bold"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Chat Messages Body */}
      <div className="flex-1 p-4 overflow-y-auto space-y-3 font-mono text-xs max-h-[380px]">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl p-3 text-xs leading-relaxed ${
                msg.sender === 'user'
                  ? 'bg-cyan-700 text-white rounded-tr-none shadow-xs font-semibold'
                  : msg.sender === 'system'
                  ? 'bg-amber-100 dark:bg-amber-950/50 text-amber-950 dark:text-amber-200 border border-amber-300 dark:border-amber-500/30'
                  : 'bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 text-slate-950 dark:text-slate-200 rounded-tl-none shadow-xs font-medium'
              }`}
            >
              <p className="whitespace-pre-line">{msg.content}</p>

              {/* Data points HUD within assistant bubble */}
              {msg.dataPoints && msg.dataPoints.length > 0 && (
                <div className="mt-2.5 pt-2 border-t border-slate-300 dark:border-slate-800 grid grid-cols-1 gap-1">
                  {msg.dataPoints.map((dp, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-white dark:bg-black/40 border border-slate-200 dark:border-transparent px-2 py-1 rounded">
                      <span className="text-slate-700 dark:text-slate-400 font-semibold">{dp.label}:</span>
                      <strong className="text-cyan-900 dark:text-cyan-300 font-bold">{dp.value}</strong>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <span className="text-[9px] text-slate-600 dark:text-slate-500 mt-1 px-1 font-semibold">{msg.timestamp}</span>
          </div>
        ))}

        {isTyping && (
          <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400 text-xs font-mono font-medium">
            <span className="w-2 h-2 rounded-full bg-cyan-600 animate-bounce"></span>
            <span className="w-2 h-2 rounded-full bg-cyan-600 animate-bounce [animation-delay:0.2s]"></span>
            <span className="w-2 h-2 rounded-full bg-cyan-600 animate-bounce [animation-delay:0.4s]"></span>
            <span className="text-[10px]">Analyzing multi-source telemetry...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Row */}
      <div className="p-3 bg-slate-100 dark:bg-slate-900/90 border-t border-slate-300 dark:border-slate-800">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            placeholder="Ask cyclone intelligence query..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            className="flex-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-950 dark:text-white font-mono placeholder:text-slate-500 dark:placeholder:text-slate-500 focus:outline-none focus:border-cyan-600 font-medium"
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || isTyping}
            className="p-2 rounded-xl bg-cyan-700 hover:bg-cyan-600 disabled:opacity-40 text-white transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
        <div className="mt-1.5 flex items-center justify-between text-[9px] font-mono text-slate-600 dark:text-slate-500 font-medium">
          <span>Target: POST /api/chat</span>
          <span>Prototype Simulation</span>
        </div>
      </div>
    </div>
  );
};
