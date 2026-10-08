import React, { useState } from 'react';
import { Search, Image as ImageIcon, FileText, ArrowRight } from 'lucide-react';

export const CrossModalSearch: React.FC = () => {
  const [searchMode, setSearchMode] = useState<'text-fig' | 'fig-paper' | 'cap-fig'>('text-fig');
  const [searchQuery, setSearchQuery] = useState('Accuracy comparison between ResNet and Vision Transformers');

  const searchResults = [
    {
      id: 'fig_03',
      figure_title: 'Figure 3: Top-1 Accuracy Curves on CIFAR-10',
      paper_title: 'Deep Residual Learning vs. Vision Transformers for Visual Recognition',
      authors: 'He et al. & Dosovitskiy et al. (arXiv 2026)',
      relevance_score: 0.942,
      caption_snippet: 'Comparative top-1 accuracy curves for ResNet-50 and ViT-Base across 300 training epochs on CIFAR-10...',
      entities: ['ResNet-50', 'ViT-Base', 'CIFAR-10', 'Top-1 Accuracy', '88.1%']
    },
    {
      id: 'fig_01',
      figure_title: 'Figure 1: Architecture Diagram of Multi-Modal Vision Encoder',
      paper_title: 'Multi-Modal Scientific Visualization Understanding with Hierarchical Grounding',
      authors: 'Academic Research Prototype (2026)',
      relevance_score: 0.885,
      caption_snippet: 'End-to-End architecture connecting 2D plot layout extraction to SciBERT entity grounding graphs...',
      entities: ['Hierarchical RAG', 'SciBERT', 'NetworkX', 'FAISS', 'SCQ Metric']
    }
  ];

  return (
    <div className="max-w-6xl mx-auto px-6 py-10 space-y-8">
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold text-slate-100 flex items-center justify-center gap-2">
          <Search className="w-5 h-5 text-cyan-400" />
          Multimodal Cross-Modal Search
        </h2>
        <p className="text-xs text-slate-400">
          Search figures, captions, and paper context across text, vision, and entity graph embeddings.
        </p>
      </div>

      {/* Mode Switcher */}
      <div className="flex justify-center gap-2">
        {[
          { id: 'text-fig', label: 'Text → Figure Search' },
          { id: 'fig-paper', label: 'Figure → Paper Search' },
          { id: 'cap-fig', label: 'Caption → Figure Search' }
        ].map((mode) => (
          <button
            key={mode.id}
            onClick={() => setSearchMode(mode.id as any)}
            className={`px-4 py-2 rounded-xl text-xs font-mono transition-all border ${
              searchMode === mode.id
                ? 'bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 text-cyan-300 border-cyan-500/40 shadow-lg font-semibold'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            {mode.label}
          </button>
        ))}
      </div>

      {/* Search Input Bar */}
      <div className="glass-panel p-3 rounded-2xl border border-slate-800 flex items-center gap-3 shadow-2xl">
        <Search className="w-5 h-5 text-cyan-400 ml-2" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Enter natural language query or paste figure text..."
          className="flex-1 bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none font-mono"
        />
        <button className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 transition-all">
          Search
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Results Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400 border-b border-slate-800 pb-2">
          <span>Found {searchResults.length} Multimodal Search Results</span>
          <span>Embedding Similarity: Cosine / CLIP ViT-Base</span>
        </div>

        {searchResults.map((res) => (
          <div key={res.id} className="glass-panel p-5 rounded-xl border border-slate-800 hover:border-cyan-500/40 transition-all flex flex-col md:flex-row gap-5 shadow-lg">
            {/* Visual Thumbnail */}
            <div className="w-full md:w-56 h-36 bg-slate-950 rounded-lg border border-slate-800 flex items-center justify-center relative overflow-hidden flex-shrink-0">
              <svg width="100%" height="100%" viewBox="0 0 200 120" className="opacity-80">
                <line x1="20" y1="20" x2="20" y2="100" stroke="#334155" strokeWidth="2" />
                <line x1="20" y1="100" x2="180" y2="100" stroke="#334155" strokeWidth="2" />
                <path d="M 20 80 Q 70 50, 110 40 T 180 20" fill="none" stroke="#38bdf8" strokeWidth="2.5" />
                <path d="M 20 90 Q 70 80, 110 50 T 180 15" fill="none" stroke="#818cf8" strokeWidth="2.5" />
              </svg>
              <span className="absolute top-2 left-2 px-2 py-0.5 rounded bg-slate-900/90 text-[10px] font-mono text-cyan-400 border border-slate-800">
                {(res.relevance_score * 100).toFixed(1)}% Match
              </span>
            </div>

            {/* Info */}
            <div className="flex-1 space-y-2">
              <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-cyan-400" />
                {res.figure_title}
              </h3>
              <p className="text-xs text-cyan-400 font-mono flex items-center gap-1">
                <FileText className="w-3.5 h-3.5 text-indigo-400" />
                {res.paper_title}
              </p>
              <p className="text-xs text-slate-300 italic font-sans">
                "{res.caption_snippet}"
              </p>

              <div className="flex flex-wrap gap-1.5 pt-2 font-mono">
                {res.entities.map((ent, idx) => (
                  <span key={idx} className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[10px] text-slate-300">
                    {ent}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
