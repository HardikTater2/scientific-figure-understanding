import React, { useState } from 'react';
import { ShieldCheck, CheckCircle2, AlertTriangle, XCircle, Info, BarChart3 } from 'lucide-react';
import type { PipelineResult, ClaimVerification } from '../types';

interface VerificationPanelProps {
  pipelineResult: PipelineResult;
}

export const VerificationPanel: React.FC<VerificationPanelProps> = ({ pipelineResult }) => {
  const { verification_results, scq_evaluation } = pipelineResult;
  const [selectedClaim, setSelectedClaim] = useState<ClaimVerification | null>(
    verification_results.claims_verification[0] || null
  );

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'SUPPORTED':
        return (
          <span className="px-2.5 py-1 rounded-full bg-emerald-950 border border-emerald-500/30 text-emerald-300 font-mono text-[10px] flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" /> SUPPORTED
          </span>
        );
      case 'UNKNOWN':
        return (
          <span className="px-2.5 py-1 rounded-full bg-amber-950 border border-amber-500/30 text-amber-300 font-mono text-[10px] flex items-center gap-1">
            <AlertTriangle className="w-3 h-3 text-amber-400" /> UNKNOWN
          </span>
        );
      case 'CONTRADICTED':
        return (
          <span className="px-2.5 py-1 rounded-full bg-rose-950 border border-rose-500/30 text-rose-300 font-mono text-[10px] flex items-center gap-1">
            <XCircle className="w-3 h-3 text-rose-400" /> CONTRADICTED
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* SCQ Composite Score Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
          <div className="flex items-center gap-2 text-cyan-400 font-semibold text-xs font-mono">
            <BarChart3 className="w-4 h-4" />
            Scientific Caption Quality (SCQ) Composite Metric
          </div>
          <span className="text-xl font-bold text-cyan-400 font-mono">
            {(scq_evaluation.scq_composite_score * 100).toFixed(1)} / 100
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
          <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Entity Coverage (w=0.30)</span>
            <span className="text-cyan-300 font-bold text-sm">
              {(scq_evaluation.breakdown.entity_coverage * 100).toFixed(0)}%
            </span>
          </div>
          <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Technical Correctness (w=0.30)</span>
            <span className="text-emerald-300 font-bold text-sm">
              {(scq_evaluation.breakdown.technical_correctness * 100).toFixed(0)}%
            </span>
          </div>
          <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Reconstruction Fidelity (w=0.20)</span>
            <span className="text-indigo-300 font-bold text-sm">
              {(scq_evaluation.breakdown.reconstruction_fidelity * 100).toFixed(0)}%
            </span>
          </div>
          <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Retrieval Accuracy (w=0.20)</span>
            <span className="text-purple-300 font-bold text-sm">
              {(scq_evaluation.breakdown.retrieval_accuracy * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Claim Verification Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800 text-xs font-mono text-slate-200">
          <span className="font-semibold flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Caption Factual Claim Entailment Verification
          </span>
          <span className="text-[11px] text-slate-400">
            {verification_results.claims_verification.length} Claims Checked
          </span>
        </div>

        <div className="space-y-2">
          {verification_results.claims_verification.map((claimObj, idx) => (
            <div
              key={idx}
              onClick={() => setSelectedClaim(claimObj)}
              className={`p-3 rounded-lg border text-xs cursor-pointer transition-all ${
                selectedClaim?.claim === claimObj.claim
                  ? 'bg-slate-800/90 border-cyan-500/40 shadow-md'
                  : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <p className="text-slate-200 font-sans leading-relaxed">
                  "{claimObj.claim}"
                </p>
                {getStatusBadge(claimObj.status)}
              </div>
            </div>
          ))}
        </div>

        {/* Selected Claim Evidence Detail */}
        {selectedClaim && (
          <div className="mt-4 p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono space-y-2">
            <div className="flex items-center gap-2 text-cyan-400 font-semibold">
              <Info className="w-3.5 h-3.5" /> Supporting Evidence for Claim:
            </div>
            <p className="text-slate-300 italic font-sans text-xs">
              "{selectedClaim.evidence_snippet}"
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
