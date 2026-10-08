import React from 'react';
import { motion } from 'framer-motion';
import { FileText, Image as ImageIcon, Layers, Cpu, Network, ArrowRight, Sparkles, BookOpen, ShieldCheck } from 'lucide-react';

interface LandingPageProps {
  onStartAnalysis: () => void;
  onExploreDemo: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onStartAnalysis, onExploreDemo }) => {
  const pipelineSteps = [
    { id: 1, title: 'PDF Ingestion', icon: FileText, desc: 'Hierarchy parsing (Paper -> Section -> Para -> Fig)' },
    { id: 2, title: 'Figure OCR & Layout', icon: ImageIcon, desc: 'OpenCV detection of axes, legends & chart regions' },
    { id: 3, title: '4-Level Context RAG', icon: Layers, desc: 'FAISS indexing across paper, section, para & entity' },
    { id: 4, title: 'Scientific NER', icon: Cpu, desc: 'SciBERT extraction of methods, metrics & datasets' },
    { id: 5, title: 'Entity Grounding Graph', icon: Network, desc: 'Heterogeneous NetworkX node & edge construction' },
    { id: 6, title: 'Grounded Captioning & SCQ', icon: ShieldCheck, desc: 'VLM generation with claim evidence verification' }
  ];

  const capabilities = [
    {
      title: '4-Level Hierarchical Context',
      desc: 'Retrieves contextual evidence simultaneously at paper, section, paragraph, and entity granularities with configurable scoring weights.',
      icon: Layers,
      color: 'from-cyan-500/20 to-blue-500/10 border-cyan-500/30'
    },
    {
      title: 'Scientific NER & Relation Extraction',
      desc: 'SciBERT-backed abstraction layer identifying METHOD, MODEL, DATASET, METRIC, VARIABLE, and ACHIEVES/EVALUATED_ON triples.',
      icon: Cpu,
      color: 'from-indigo-500/20 to-purple-500/10 border-indigo-500/30'
    },
    {
      title: 'Figure Element Layout Analysis',
      desc: 'Detects and bounds visual components including X/Y axes, chart legends, data curves, and chart canvases in 2D scientific plots.',
      icon: ImageIcon,
      color: 'from-emerald-500/20 to-teal-500/10 border-emerald-500/30'
    },
    {
      title: 'Heterogeneous Entity Grounding Graph',
      desc: 'NetworkX graph topology connecting detected figure layout elements directly to paper entities via REPRESENTS and CORRESPONDS_TO edges.',
      icon: Network,
      color: 'from-purple-500/20 to-pink-500/10 border-purple-500/30'
    },
    {
      title: 'Cross-Modal Tri-Modal Retrieval',
      desc: 'Shared embedding space supporting circular search across Text -> Figure, Figure -> Paper, and Caption -> Figure queries.',
      icon: Sparkles,
      color: 'from-amber-500/20 to-orange-500/10 border-amber-500/30'
    },
    {
      title: 'Grounded Captions & SCQ Verification',
      desc: 'Evidence-backed caption generation with claim-level NLI entailment checking and composite Scientific Caption Quality (SCQ) scoring.',
      icon: ShieldCheck,
      color: 'from-blue-500/20 to-cyan-500/10 border-blue-500/30'
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 bg-grid-pattern pb-20">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-6 pt-16 pb-12 text-center relative">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono text-cyan-400 mb-6 shadow-xl">
            <Sparkles className="w-3.5 h-3.5" />
            Academic Research Prototype • Multi-Modal Scientific Visualization Understanding
          </div>

          <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-white max-w-4xl mx-auto leading-tight">
            Multi-Modal Scientific Visualization Understanding with{' '}
            <span className="bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-400 bg-clip-text text-transparent">
              Hierarchical Grounding
            </span>
          </h1>

          <p className="mt-6 text-base md:text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Bridging the gap between scientific figures, surrounding paper context, and entity grounding graphs for interpretable multimodal retrieval and grounded caption generation.
          </p>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={onStartAnalysis}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-medium text-sm shadow-lg shadow-cyan-500/25 flex items-center gap-2 transition-all group cursor-pointer"
            >
              Analyze a Scientific Paper
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
            <button
              onClick={onExploreDemo}
              className="px-6 py-3 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-200 font-medium text-sm flex items-center gap-2 transition-all cursor-pointer"
            >
              <BookOpen className="w-4 h-4 text-cyan-400" />
              Explore Interactive Demo
            </button>
          </div>
        </motion.div>
      </div>

      {/* Interactive Visual Pipeline Workflow */}
      <div className="max-w-7xl mx-auto px-6 my-12">
        <div className="glass-panel rounded-2xl p-8 border border-slate-800 shadow-2xl">
          <div className="text-center mb-8">
            <h2 className="text-xl font-semibold text-slate-100 font-mono tracking-tight">
              ARCHITECTURE & DATA RETRIEVAL FLOW
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              End-to-End Pipeline from Document Ingestion to Evidence-Grounded Caption Verification
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 relative">
            {pipelineSteps.map((step) => {
              const Icon = step.icon;
              return (
                <div
                  key={step.id}
                  className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col items-center text-center hover:border-cyan-500/40 transition-all duration-300 relative group shadow-md"
                >
                  <div className="w-10 h-10 rounded-xl bg-slate-800/80 border border-slate-700/50 flex items-center justify-center text-cyan-400 mb-3 group-hover:scale-110 transition-transform">
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-[10px] font-mono text-cyan-400 px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/20 mb-1">
                    STAGE 0{step.id}
                  </span>
                  <h3 className="text-xs font-semibold text-slate-200 mb-1">{step.title}</h3>
                  <p className="text-[11px] text-slate-400 leading-tight">{step.desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Capabilities Grid */}
      <div className="max-w-7xl mx-auto px-6 mt-16">
        <div className="text-center mb-10">
          <h2 className="text-2xl font-bold text-slate-100">Core Architecture & Capabilities</h2>
          <p className="text-sm text-slate-400 mt-2">Modular services for scientific paper & figure understanding</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {capabilities.map((cap, i) => {
            const Icon = cap.icon;
            return (
              <div
                key={i}
                className={`bg-gradient-to-br ${cap.color} backdrop-blur-sm rounded-xl p-6 border transition-all duration-300 hover:-translate-y-1 shadow-lg`}
              >
                <div className="w-10 h-10 rounded-xl bg-slate-900/90 border border-slate-800 flex items-center justify-center text-cyan-400 mb-4">
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="text-base font-semibold text-slate-100 mb-2">{cap.title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{cap.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
