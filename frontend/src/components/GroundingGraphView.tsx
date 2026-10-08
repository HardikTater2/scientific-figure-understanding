import React, { useState } from 'react';
import { Network, ZoomIn, ZoomOut, RotateCcw, Info } from 'lucide-react';
import type { GroundingGraphSummary, GraphNode } from '../types';

interface GroundingGraphViewProps {
  graphSummary: GroundingGraphSummary;
}

export const GroundingGraphView: React.FC<GroundingGraphViewProps> = ({ graphSummary }) => {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(graphSummary.nodes[0] || null);
  const [zoom, setZoom] = useState(1);

  const nodesWithCoords = graphSummary.nodes.map((node, i) => {
    const total = graphSummary.nodes.length;
    const angle = (i / total) * 2 * Math.PI;
    const radius = 170;
    const centerX = 300;
    const centerY = 200;
    return {
      ...node,
      cx: centerX + radius * Math.cos(angle),
      cy: centerY + radius * Math.sin(angle)
    };
  });

  const nodeMap = new Map(nodesWithCoords.map((n) => [n.id, n]));

  const connectedNodeIds = new Set<string>();
  if (selectedNode) {
    connectedNodeIds.add(selectedNode.id);
    graphSummary.edges.forEach((edge) => {
      if (edge.source === selectedNode.id) connectedNodeIds.add(edge.target);
      if (edge.target === selectedNode.id) connectedNodeIds.add(edge.source);
    });
  }

  const getNodeColor = (node: GraphNode) => {
    if (node.node_type === 'FIGURE_ELEMENT') return '#38bdf8';
    if (node.entity_type === 'METHOD' || node.entity_type === 'MODEL') return '#818cf8';
    if (node.entity_type === 'DATASET') return '#a855f7';
    if (node.entity_type === 'METRIC' || node.entity_type === 'METRIC_VALUE') return '#34d399';
    return '#f59e0b';
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col h-full shadow-xl">
      {/* Header bar */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800 text-xs font-mono">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-cyan-400" />
          <span className="font-semibold text-slate-200">Heterogeneous Entity Grounding Graph</span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500/30 text-cyan-300">
            {graphSummary.num_nodes} Nodes • {graphSummary.num_edges} Edges
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button onClick={() => setZoom(Math.min(zoom + 0.2, 1.8))} className="p-1 rounded bg-slate-800 text-slate-300">
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

      {/* Graph Visual Canvas */}
      <div className="relative flex-1 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center p-4 min-h-[360px]">
        <div style={{ transform: `scale(${zoom})` }} className="transition-transform duration-200">
          <svg width="600" height="400" className="bg-slate-950">
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
              </marker>
              <marker id="arrow-active" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#38bdf8" />
              </marker>
            </defs>

            {/* Edges */}
            {graphSummary.edges.map((edge, idx) => {
              const srcNode = nodeMap.get(edge.source);
              const tgtNode = nodeMap.get(edge.target);
              if (!srcNode || !tgtNode) return null;

              const isConnected = selectedNode && (edge.source === selectedNode.id || edge.target === selectedNode.id);

              const midX = (srcNode.cx + tgtNode.cx) / 2;
              const midY = (srcNode.cy + tgtNode.cy) / 2;

              return (
                <g key={idx}>
                  <line
                    x1={srcNode.cx}
                    y1={srcNode.cy}
                    x2={tgtNode.cx}
                    y2={tgtNode.cy}
                    stroke={isConnected ? '#38bdf8' : '#334155'}
                    strokeWidth={isConnected ? 2.5 : 1.2}
                    strokeDasharray={edge.relation.includes('REPRESENTS') ? '4 2' : undefined}
                    markerEnd={isConnected ? 'url(#arrow-active)' : 'url(#arrow)'}
                  />
                  <text
                    x={midX}
                    y={midY - 4}
                    fill={isConnected ? '#38bdf8' : '#64748b'}
                    fontSize="9"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    {edge.relation}
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {nodesWithCoords.map((node) => {
              const isSelected = selectedNode?.id === node.id;
              const isConnected = connectedNodeIds.has(node.id);
              const nodeColor = getNodeColor(node);

              return (
                <g
                  key={node.id}
                  onClick={() => setSelectedNode(node)}
                  className="cursor-pointer transition-all duration-200"
                >
                  <circle
                    cx={node.cx}
                    cy={node.cy}
                    r={isSelected ? 16 : isConnected ? 13 : 10}
                    fill={nodeColor}
                    fillOpacity={isConnected ? 0.9 : 0.4}
                    stroke={nodeColor}
                    strokeWidth={isSelected ? 3 : 1.5}
                    className="hover:scale-125 transition-transform"
                  />
                  <text
                    x={node.cx}
                    y={node.cy + 24}
                    fill={isConnected ? '#f1f5f9' : '#94a3b8'}
                    fontSize="10"
                    fontFamily="monospace"
                    fontWeight={isSelected ? 'bold' : 'normal'}
                    textAnchor="middle"
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Selected Node Details Info */}
      {selectedNode && (
        <div className="mt-3 p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono">
          <div className="flex items-center justify-between text-slate-200 font-semibold mb-1">
            <span className="flex items-center gap-1.5 text-cyan-400">
              <Info className="w-3.5 h-3.5" /> Selected Node: {selectedNode.label}
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">
              Type: {selectedNode.node_type} ({selectedNode.element_type || selectedNode.entity_type || 'Node'})
            </span>
          </div>
          <p className="text-slate-400 text-[11px]">
            ID: {selectedNode.id} • Connected Edges: {Array.from(connectedNodeIds).length - 1} graph connections
          </p>
        </div>
      )}
    </div>
  );
};
