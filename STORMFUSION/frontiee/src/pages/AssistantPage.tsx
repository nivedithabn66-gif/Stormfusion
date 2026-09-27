import React from 'react';
import { Bot, Sparkles, Database, ShieldAlert, Cpu, Layers } from 'lucide-react';
import { CycloneAIChatbot } from '../components/CycloneAIChatbot';
import { DemoModeBadge } from '../components/DemoModeBadge';

export const AssistantPage: React.FC = () => {
  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-[1400px] mx-auto font-mono">
      {/* Header */}
      <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 dark:bg-cyan-500/15 border border-cyan-500/30 text-cyan-600 dark:text-cyan-400">
            <Bot className="w-7 h-7" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-wide text-slate-900 dark:text-white uppercase">
              STORMFUSION CONTEXTUAL ASSISTANT
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Conversational decision-support interface directly grounded in real-time system state
            </p>
          </div>
        </div>

        <DemoModeBadge subtle />
      </div>

      {/* Main Container */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side: System Grounding Facts & Prompt Suggestions */}
        <div className="lg:col-span-4 space-y-4 text-xs">
          <div className="command-card rounded-2xl p-5 border border-slate-200 dark:border-slate-800 space-y-3">
            <div className="flex items-center gap-2 text-slate-900 dark:text-white font-bold uppercase pb-2 border-b border-slate-200 dark:border-slate-800">
              <Database className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
              Live Telemetry Context
            </div>
            <div className="space-y-2 text-slate-700 dark:text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-500">Active System:</span>
                <strong className="text-slate-900 dark:text-white">CYCLONE DEMO</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Location:</span>
                <strong className="text-slate-900 dark:text-white">14.52° N, 87.21° E</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Intensity:</span>
                <strong className="text-rose-600 dark:text-rose-400">82 kt (152 km/h)</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Pressure:</span>
                <strong className="text-slate-900 dark:text-white">965 hPa</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Movement:</span>
                <strong className="text-sky-600 dark:text-sky-300">NW @ 14 km/h</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Landfall Target:</span>
                <strong className="text-amber-700 dark:text-amber-400">+72h (North AP)</strong>
              </div>
            </div>
          </div>

          <div className="command-card rounded-2xl p-5 border border-purple-200 dark:border-purple-500/30 space-y-3 bg-purple-50/50 dark:bg-purple-950/20">
            <div className="flex items-center gap-2 text-purple-800 dark:text-purple-300 font-bold uppercase pb-2 border-b border-purple-200 dark:border-purple-500/30">
              <Cpu className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              Grounded AI Architecture
            </div>
            <p className="text-[11px] text-slate-700 dark:text-slate-300 leading-relaxed">
              Unlike generic LLMs that may hallucinate meteorological tracks, the StormFusion Assistant is strictly bounded to the structured JSON outputs from the ConvLSTM track engine and GIS risk indexes.
            </p>
            <div className="text-[10px] text-purple-700 dark:text-purple-400">
              FastAPI Endpoint: <code>POST /api/chat</code>
            </div>
          </div>
        </div>

        {/* Right Side: Embedded Chatbot Interface */}
        <div className="lg:col-span-8 h-[650px]">
          <CycloneAIChatbot isFloating={false} />
        </div>
      </div>
    </div>
  );
};
