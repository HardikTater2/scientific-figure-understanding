import React, { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Layers, Eye, Sparkles, Box, Orbit } from 'lucide-react';
import type { FigureElement, ThreeDBranchAnalysis } from '../types';

interface FigureViewerProps {
  elements: FigureElement[];
  threedAnalysis?: ThreeDBranchAnalysis;
  onSelectElement?: (element: FigureElement) => void;
}

export const FigureViewer: React.FC<FigureViewerProps> = ({ elements, threedAnalysis, onSelectElement }) => {
  const [zoom, setZoom] = useState(1);
  const [selectedElement, setSelectedElement] = useState<FigureElement | null>(elements[1] || null);
  const [showOverlays, setShowOverlays] = useState(true);
  const [viewMode, setViewMode] = useState<'2d' | '3d'>(
    threedAnalysis?.['3d_detection']?.is_3d ? '3d' : '2d'
  );

  const is3D = threedAnalysis?.['3d_detection']?.is_3d || false;
  const visType = threedAnalysis?.['3d_detection']?.visualization_type || '2D_PLOT';

  const handleElementClick = (elem: FigureElement) => {
    setSelectedElement(elem);
    if (onSelectElement) onSelectElement(elem);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col h-full shadow-xl">
      {/* Controls & Mode Header */}
      <div className="flex flex-wrap items-center justify-between pb-3 border-b border-slate-800 text-xs gap-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-cyan-400 font-semibold text-xs flex items-center gap-1.5">
            {is3D ? <Box className="w-4 h-4 text-emerald-400" /> : <Layers className="w-3.5 h-3.5" />}
            {is3D ? '3D Scientific Visualization Canvas' : '2D Scientific Plot Canvas'}
          </span>
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${
            is3D ? 'bg-emerald-950 text-emerald-300 border border-emerald-500/30' : 'bg-slate-800 text-slate-400'
          }`}>
            {visType}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Mode Switcher if 3D is available */}
          <div className="flex bg-slate-950 p-0.5 rounded-lg border border-slate-800">
            <button
              onClick={() => setViewMode('2d')}
              className={`px-2.5 py-1 rounded text-[10px] font-mono transition-all ${
                viewMode === '2d' ? 'bg-cyan-950 text-cyan-300 font-semibold border border-cyan-500/30' : 'text-slate-400'
              }`}
            >
              2D Layout
            </button>
            <button
              onClick={() => setViewMode('3d')}
              className={`px-2.5 py-1 rounded text-[10px] font-mono transition-all flex items-center gap-1 ${
                viewMode === '3d' ? 'bg-emerald-950 text-emerald-300 font-semibold border border-emerald-500/30' : 'text-slate-400'
              }`}
            >
              <Orbit className="w-3 h-3" /> 3D PointNet
            </button>
          </div>

          <button
            onClick={() => setShowOverlays(!showOverlays)}
            className={`px-2.5 py-1 rounded text-[11px] font-medium flex items-center gap-1 border transition-all ${
              showOverlays ? 'bg-cyan-950/80 border-cyan-500/40 text-cyan-300' : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            <Eye className="w-3 h-3" /> Overlays {showOverlays ? 'ON' : 'OFF'}
          </button>
          <button onClick={() => setZoom(Math.min(zoom + 0.2, 2.0))} className="p-1 rounded bg-slate-800 text-slate-300">
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setZoom(Math.max(zoom - 0.2, 0.6))} className="p-1 rounded bg-slate-800 text-slate-300">
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setZoom(1.0)} className="p-1 rounded bg-slate-800 text-slate-300">
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Visual Canvas Area */}
      <div className="relative flex-1 bg-slate-950/90 rounded-lg overflow-hidden flex items-center justify-center p-6 min-h-[340px]">
        <div className="relative transition-transform duration-200" style={{ transform: `scale(${zoom})` }}>
          {viewMode === '2d' ? (
            /* 2D SVG Plot Canvas */
            <svg width="600" height="340" className="border border-slate-800 rounded bg-slate-900 shadow-inner">
              <line x1="70" y1="50" x2="70" y2="280" stroke="#334155" strokeWidth="2" />
              <line x1="70" y1="280" x2="550" y2="280" stroke="#334155" strokeWidth="2" />
              {[100, 150, 200, 250].map((y) => (
                <line key={y} x1="70" y1={y} x2="550" y2={y} stroke="#1e293b" strokeDasharray="4 4" />
              ))}
              <path d="M 70 240 Q 150 180, 250 140 T 450 100 T 550 95" fill="none" stroke="#38bdf8" strokeWidth="3" />
              <path d="M 70 270 Q 150 250, 250 180 T 450 85 T 550 70" fill="none" stroke="#818cf8" strokeWidth="3" />

              <text x="280" y="315" fill="#94a3b8" fontSize="11" fontFamily="monospace">Training Epochs (0 - 300)</text>
              <text x="15" y="170" fill="#94a3b8" fontSize="11" fontFamily="monospace" transform="rotate(-90 25,170)">Top-1 Accuracy (%)</text>

              {showOverlays && (
                <>
                  <rect
                    x="10" y="40" width="45" height="250"
                    fill={selectedElement?.id === 'axis_y' ? 'rgba(56, 189, 248, 0.25)' : 'rgba(56, 189, 248, 0.08)'}
                    stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="3 3"
                    className="cursor-pointer hover:fill-cyan-500/20"
                    onClick={() => handleElementClick(elements[1] || { id: 'axis_y', type: 'AXIS', text: 'Y-Axis: Top-1 Accuracy' })}
                  />
                  <rect
                    x="65" y="285" width="490" height="40"
                    fill={selectedElement?.id === 'axis_x' ? 'rgba(56, 189, 248, 0.25)' : 'rgba(56, 189, 248, 0.08)'}
                    stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="3 3"
                    className="cursor-pointer hover:fill-cyan-500/20"
                    onClick={() => handleElementClick(elements[0] || { id: 'axis_x', type: 'AXIS', text: 'X-Axis: Epochs' })}
                  />
                  <rect
                    x="410" y="55" width="135" height="60"
                    fill={selectedElement?.id === 'legend_1' ? 'rgba(129, 140, 248, 0.25)' : 'rgba(129, 140, 248, 0.08)'}
                    stroke="#818cf8" strokeWidth="1.5" strokeDasharray="3 3"
                    className="cursor-pointer hover:fill-indigo-500/20"
                    onClick={() => handleElementClick(elements[2] || { id: 'legend_1', type: 'LEGEND', text: 'Legend: ResNet & ViT' })}
                  />
                </>
              )}
            </svg>
          ) : (
            /* 3D Point-Cloud Perspective Canvas */
            <svg width="600" height="340" className="border border-emerald-500/30 rounded bg-slate-950 shadow-inner">
              <text x="210" y="30" fill="#34d399" fontSize="12" fontFamily="monospace" fontWeight="bold">
                AlphaFold 3D Protein Point-Cloud Mesh
              </text>

              {/* 3D Point Cloud Mesh Representation */}
              <ellipse cx="300" cy="180" rx="160" ry="100" fill="none" stroke="#059669" strokeWidth="1.5" strokeDasharray="4 4" />
              <ellipse cx="300" cy="180" rx="100" ry="140" fill="none" stroke="#10b981" strokeWidth="1.5" strokeDasharray="4 4" />
              
              {/* Point Cloud Atomic Nodes */}
              {Array.from({ length: 40 }).map((_, i) => {
                const angle = (i / 40) * 2 * Math.PI;
                const r = 80 + (i % 5) * 12;
                const cx = 300 + r * Math.cos(angle);
                const cy = 180 + (r * 0.6) * Math.sin(angle);
                const plddt = 85 + (i % 15);
                const color = plddt > 90 ? '#34d399' : plddt > 80 ? '#38bdf8' : '#f59e0b';

                return (
                  <g key={i}>
                    <circle cx={cx} cy={cy} r={i % 3 === 0 ? 5 : 3.5} fill={color} fillOpacity="0.85" />
                    {i % 4 === 0 && (
                      <line x1={cx} y1={cy} x2={300} y2={180} stroke={color} strokeWidth="0.8" opacity="0.3" />
                    )}
                  </g>
                );
              })}

              <rect x="420" y="240" width="160" height="85" fill="rgba(15, 23, 42, 0.9)" stroke="#34d399" strokeWidth="1" rx="6" />
              <text x="430" y="260" fill="#34d399" fontSize="10" fontFamily="monospace" fontWeight="bold">PointNet 3D Feature Norm</text>
              <text x="430" y="280" fill="#cbd5e1" fontSize="10" fontFamily="monospace">Dimensions: 1024 x 3</text>
              <text x="430" y="300" fill="#cbd5e1" fontSize="10" fontFamily="monospace">pLDDT Score &gt; 90 (Cyan)</text>
            </svg>
          )}
        </div>
      </div>

      {/* Interactive Grounding Trace Flow Panel */}
      {selectedElement && (
        <div className="mt-3 p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs font-mono">
          <div className="flex items-center gap-2 text-cyan-400 font-semibold mb-1.5">
            <Sparkles className="w-3.5 h-3.5" />
            Interactive Grounding Trace for [{selectedElement.type}]: "{selectedElement.text}"
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300">
            <span className="px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500/30 text-cyan-300">
              Element: {selectedElement.id}
            </span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-indigo-950 border border-indigo-500/30 text-indigo-300">
              Metric: Top-1 Accuracy / pLDDT
            </span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-purple-950 border border-purple-500/30 text-purple-300">
              Paragraph Context
            </span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-emerald-950 border border-emerald-500/30 text-emerald-300">
              Method: ResNet / ViT / AlphaFold
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
