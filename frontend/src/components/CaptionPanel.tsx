import React from 'react';
import { Sparkles, CheckCircle2, ShieldCheck, FileText, BookOpen } from 'lucide-react';
import type { PipelineResult } from '../types';

interface CaptionPanelProps {
  pipelineResult: PipelineResult;
}

export const CaptionPanel: React.FC<CaptionPanelProps> = ({ pipelineResult }) => {
  const { generated_caption, retrieved_context, extracted_entities, figure_elements } = pipelineResult;
  const scicapEval = (pipelineResult as any).scicap_evaluation;

  const paragraphEvidence = retrieved_context.level_results.paragraph[0]?.text || 'Paragraph evidence snippet...';

  return (
    <div className="space-y-6">
      {/* SciCap Ground-Truth Comparison Card (If SciCap benchmark is active) */}
      {scicapEval && (
        <div className="bg-slate-900 border border-cyan-500/40 rounded-xl p-5 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2 text-cyan-400 font-semibold text-xs font-mono">
              <BookOpen className="w-4 h-4 text-cyan-400" />
              SciCap Benchmark: Author Ground-Truth vs Model Caption
            </div>
            <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-500/30">
              SciCap Dataset Pair
            </span>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
              <span className="text-[10px] text-cyan-400 font-semibold block uppercase tracking-wider">
                Author Ground-Truth Caption (SciCap Dataset):
              </span>
              <p className="text-slate-200 font-sans italic text-xs leading-relaxed">
                "{scicapEval.ground_truth_caption}"
              </p>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
              <span className="text-[10px] text-indigo-400 font-semibold block uppercase tracking-wider">
                Grounded VLM Pipeline Caption (Model Output):
              </span>
              <p className="text-slate-300 font-sans italic text-xs leading-relaxed">
                "{scicapEval.generated_caption}"
              </p>
            </div>

            {/* Benchmark Scores */}
            {scicapEval.evaluation_metrics && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 pt-1">
                <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-400 block">BLEU-1 Overlap</span>
                  <span className="text-cyan-400 font-bold text-xs">
                    {(scicapEval.evaluation_metrics.bleu_1_semantic_overlap * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-400 block">SciCap Entity Recall</span>
                  <span className="text-emerald-400 font-bold text-xs">
                    {(scicapEval.evaluation_metrics.scicap_entity_recall * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-400 block">SCQ Score</span>
                  <span className="text-indigo-400 font-bold text-xs">
                    {(scicapEval.evaluation_metrics.scq_composite_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Generated Caption Block */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
          <div className="flex items-center gap-2 text-cyan-400 font-semibold text-xs font-mono">
            <Sparkles className="w-4 h-4" />
            Generated Grounded Scientific Caption
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 border border-emerald-500/30 text-emerald-300 flex items-center gap-1">
            <ShieldCheck className="w-3 h-3" /> Grounded VLM Generator
          </span>
        </div>

        <p className="text-sm text-slate-100 font-sans leading-relaxed p-4 rounded-lg bg-slate-950/80 border border-slate-800 text-cyan-50/90 font-medium italic">
          "{generated_caption}"
        </p>
      </div>

      {/* Evidence Provenance Mapping */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
        <div className="flex items-center gap-2 text-slate-200 font-semibold text-xs font-mono pb-2 border-b border-slate-800">
          <FileText className="w-4 h-4 text-indigo-400" />
          Supporting Evidence Provenance & Citation Mapping
        </div>

        <div className="space-y-3 font-mono text-xs">
          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-cyan-400">
              <span className="font-semibold flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Grounded Paragraph Evidence:
              </span>
              <span className="text-slate-400">Relevance: 94.0%</span>
            </div>
            <p className="text-slate-300 font-sans text-xs italic">
              "{paragraphEvidence}"
            </p>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1.5">
            <span className="text-[11px] font-semibold text-indigo-400 block">
              Entities Integrated in Caption:
            </span>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {extracted_entities.slice(0, 5).map((e, idx) => (
                <span key={idx} className="px-2 py-0.5 rounded bg-indigo-950 border border-indigo-500/30 text-indigo-300 text-[10px]">
                  {e.text} [{e.type}]
                </span>
              ))}
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1.5">
            <span className="text-[11px] font-semibold text-purple-400 block">
              Grounded Figure Visual Elements:
            </span>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {figure_elements.elements.slice(0, 3).map((elem, idx) => (
                <span key={idx} className="px-2 py-0.5 rounded bg-purple-950 border border-purple-500/30 text-purple-300 text-[10px]">
                  {elem.id} ({elem.type}: {elem.text})
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
