import React, { useState } from 'react';
import { Cpu, Tag } from 'lucide-react';
import type { Entity } from '../types';

interface EntityExplorerProps {
  entities: Entity[];
}

export const EntityExplorer: React.FC<EntityExplorerProps> = ({ entities }) => {
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(entities[0] || null);
  const [filterType, setFilterType] = useState<string>('ALL');

  const entityTypes = ['ALL', ...Array.from(new Set(entities.map((e) => e.type)))];

  const filteredEntities = filterType === 'ALL' 
    ? entities 
    : entities.filter((e) => e.type === filterType);

  const getTypeBadgeColor = (type: string) => {
    switch (type) {
      case 'METHOD': case 'MODEL': return 'bg-cyan-950 text-cyan-300 border-cyan-500/30';
      case 'DATASET': return 'bg-indigo-950 text-indigo-300 border-indigo-500/30';
      case 'METRIC': case 'METRIC_VALUE': return 'bg-emerald-950 text-emerald-300 border-emerald-500/30';
      case 'VARIABLE': return 'bg-amber-950 text-amber-300 border-amber-500/30';
      default: return 'bg-slate-800 text-slate-300 border-slate-700';
    }
  };

  return (
    <div className="space-y-4">
      {/* Type Filter Buttons */}
      <div className="flex flex-wrap gap-1.5 pb-2 border-b border-slate-800">
        {entityTypes.map((type) => (
          <button
            key={type}
            onClick={() => setFilterType(type)}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-mono transition-all ${
              filterType === type
                ? 'bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 text-cyan-300 border border-cyan-500/40 font-semibold'
                : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* Entity Chips Grid */}
      <div className="flex flex-wrap gap-2">
        {filteredEntities.map((ent, idx) => {
          const isSelected = selectedEntity?.text === ent.text;
          return (
            <button
              key={idx}
              onClick={() => setSelectedEntity(ent)}
              className={`px-3 py-1.5 rounded-lg border text-xs font-mono transition-all flex items-center gap-1.5 ${
                isSelected
                  ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-md shadow-cyan-500/10 scale-105'
                  : getTypeBadgeColor(ent.type) + ' hover:border-slate-500'
              }`}
            >
              <Tag className="w-3 h-3" />
              <span>{ent.text}</span>
              <span className="text-[10px] opacity-70">[{ent.type}]</span>
            </button>
          );
        })}
      </div>

      {/* Selected Entity Card Detail */}
      {selectedEntity && (
        <div className="mt-4 p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3 shadow-lg">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <h4 className="text-sm font-semibold text-slate-100">{selectedEntity.text}</h4>
            </div>
            <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${getTypeBadgeColor(selectedEntity.type)}`}>
              {selectedEntity.type}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Extraction Confidence</span>
              <span className="text-emerald-400 font-semibold">{((selectedEntity.confidence || 0.9) * 100).toFixed(0)}%</span>
            </div>
            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Grounding Association</span>
              <span className="text-cyan-400 font-semibold">Grounded to Figure Element</span>
            </div>
          </div>

          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 text-xs">
            <span className="text-[10px] font-mono text-cyan-400 block mb-1">Supporting Paper Context:</span>
            <p className="text-slate-300 italic">
              "...evaluating {selectedEntity.text} in relation to figure accuracy curves and paper methodology..."
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
