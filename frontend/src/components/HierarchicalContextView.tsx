import React, { useState } from 'react';
import { ChevronRight, ChevronDown, Layers, FileText, BookOpen, Cpu } from 'lucide-react';
import type { HierarchicalRetrievalResults } from '../types';

interface HierarchicalContextViewProps {
  retrievedContext: HierarchicalRetrievalResults;
}

export const HierarchicalContextView: React.FC<HierarchicalContextViewProps> = ({ retrievedContext }) => {
  const [expandedLevels, setExpandedLevels] = useState<Record<string, boolean>>({
    paper: true,
    section: true,
    paragraph: true,
    entity: true
  });

  const toggleLevel = (level: string) => {
    setExpandedLevels((prev) => ({ ...prev, [level]: !prev[level] }));
  };

  const levelConfigs = [
    {
      id: 'paper',
      title: 'LEVEL 1: Paper-Level Context',
      icon: FileText,
      weight: 'w_paper = 0.15',
      color: 'border-cyan-500/30 text-cyan-400 bg-cyan-950/30',
      data: retrievedContext.level_results.paper
    },
    {
      id: 'section',
      title: 'LEVEL 2: Section-Level Context',
      icon: BookOpen,
      weight: 'w_section = 0.25',
      color: 'border-indigo-500/30 text-indigo-400 bg-indigo-950/30',
      data: retrievedContext.level_results.section
    },
    {
      id: 'paragraph',
      title: 'LEVEL 3: Paragraph-Level Context',
      icon: Layers,
      weight: 'w_paragraph = 0.40',
      color: 'border-purple-500/30 text-purple-400 bg-purple-950/30',
      data: retrievedContext.level_results.paragraph
    },
    {
      id: 'entity',
      title: 'LEVEL 4: Entity-Level Context',
      icon: Cpu,
      weight: 'w_entity = 0.20',
      color: 'border-emerald-500/30 text-emerald-400 bg-emerald-950/30',
      data: retrievedContext.level_results.entity
    }
  ];

  return (
    <div className="space-y-4 text-xs font-mono">
      {/* Weighted Score Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center justify-between shadow-md">
        <div>
          <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Context Weighting Formula</span>
          <span className="text-slate-200 font-semibold text-xs mt-0.5 block">
            Score = 0.15·Paper + 0.25·Section + 0.40·Para + 0.20·Entity
          </span>
        </div>
        <div className="text-right">
          <span className="text-[11px] text-slate-400 block">Weighted Relevance</span>
          <span className="text-base font-bold text-cyan-400">
            {(retrievedContext.weighted_score * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* 4-Level Collapsible Tree */}
      {levelConfigs.map((lvl) => {
        const Icon = lvl.icon;
        const isExpanded = expandedLevels[lvl.id];

        return (
          <div key={lvl.id} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
            <button
              onClick={() => toggleLevel(lvl.id)}
              className="w-full px-4 py-3 bg-slate-900 hover:bg-slate-800/80 flex items-center justify-between border-b border-slate-800/60 transition-colors"
            >
              <div className="flex items-center gap-2.5">
                {isExpanded ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
                <Icon className={`w-4 h-4 ${lvl.color.split(' ')[1]}`} />
                <span className="font-semibold text-slate-200 text-xs">{lvl.title}</span>
              </div>
              <span className={`text-[10px] px-2 py-0.5 rounded border ${lvl.color}`}>
                {lvl.weight}
              </span>
            </button>

            {isExpanded && (
              <div className="p-4 space-y-3 bg-slate-950/60">
                {lvl.data.length > 0 ? (
                  lvl.data.map((chunk, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1.5">
                      <div className="flex items-center justify-between text-[11px] text-slate-400 border-b border-slate-800/50 pb-1">
                        <span className="text-cyan-400 font-semibold">
                          Source: {chunk.section_id ? `${chunk.section_id} -> ` : ''}{chunk.paragraph_id || chunk.chunk_type}
                        </span>
                        {chunk.score && (
                          <span className="text-emerald-400 font-mono">
                            Relevance: {(chunk.score * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      <p className="text-slate-300 leading-relaxed font-sans text-xs">
                        "{chunk.text}"
                      </p>
                    </div>
                  ))
                ) : (
                  <p className="text-slate-500 italic text-center py-2">No context retrieved at this level</p>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
