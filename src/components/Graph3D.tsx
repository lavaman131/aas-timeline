import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type {
  Data as PlotlyData,
  Layout as PlotlyLayout,
  Config as PlotlyConfig,
  PlotMouseEvent,
} from 'plotly.js';

import type { TextItem } from '../utils/sampleData';
import type { FilterOptions } from './FilterControls';

interface Graph3DProps {
  data: TextItem[];
  onSelectNode: (item: TextItem | null) => void;
  filters: FilterOptions;
}

const Graph3D: React.FC<Graph3DProps> = ({ data, onSelectNode, filters }) => {
  // 1) Filter your data
  const filteredData = useMemo<TextItem[]>(() => {
    return data.filter(
      (item) =>
        filters.types.includes(item.type)
    );
  }, [data, filters]);

  // 2) Compute positions from embeddings
  const positions = useMemo<number[][]>(() => {
    if (filteredData.length === 0) {
      return [];
    }
    const embeddings = filteredData.map((item) => item.embedding_pca);
    return embeddings;
  }, [filteredData]);

  // 3) Build the Plotly “data” array
  const plotData = useMemo<PlotlyData[]>(() => {
    const x = positions.map((p) => p[0]);
    const y = positions.map((p) => p[1]);
    const z = positions.map((p) => p[2]);

    const colors = filteredData.map((item) => {
      switch (item.type) {
        case 'novel':
          return '#8B5CF6';
        case 'article':
          return '#3B82F6';
        case 'poem':
          return '#EC4899';
        default:
          return '#10B981';
      }
    });

    const hoverText = filteredData.map(
      (item) =>
        `${item.title}<br>by ${item.author} (${item.year})<br>${item.type}`,
    );

    return [
      {
        type: 'scatter3d',
        mode: 'markers',
        x,
        y,
        z,
        text: hoverText,
        hoverinfo: 'text',
        marker: {
          size: 8,
          color: colors,
          opacity: 0.8,
          symbol: 'circle',
          line: { color: '#FFFFFF', width: 1 },
        },
        name: 'Literary Works',
      },
    ];
  }, [positions, filteredData]);

  // 4) Build the Plotly layout
  const layout = useMemo<Partial<PlotlyLayout>>(() => {
    return {
      autosize: true,
      height: window.innerHeight,
      margin: { l: 0, r: 0, b: 0, t: 0 },
      paper_bgcolor: 'rgba(15, 23, 42, 0)',
      plot_bgcolor: 'rgba(15, 23, 42, 0)',
      scene: {
        xaxis: {
          showgrid: true,
          zeroline: false,
          showline: false,
          showticklabels: false,
          title: '',
          backgroundcolor: 'rgba(15, 23, 42, 0.1)',
          gridcolor: '#444',
        },
        yaxis: {
          showgrid: true,
          zeroline: false,
          showline: false,
          showticklabels: false,
          title: '',
          gridcolor: '#444',
        },
        zaxis: {
          showgrid: true,
          zeroline: false,
          showline: false,
          showticklabels: false,
          title: '',
          gridcolor: '#444',
        },
        bgcolor: 'rgba(15, 23, 42, 1)',
      },
      hovermode: 'closest',
    };
  }, []);

  // 5) Build the Plotly config
  const config = useMemo<Partial<PlotlyConfig>>(
    () => ({
      displayModeBar: false,
      responsive: true,
    }),
    [],
  );

  // 6) Handle clicks
  const handlePointClick = (e: PlotMouseEvent) => {
    const pts = e.points;
    if (pts && pts.length > 0) {
      // pointIndex is always defined for scatter3d
      const idx = pts[0].pointIndex!;
      onSelectNode(filteredData[idx]);
    }
  };

  return (
    <div className="w-full h-full">
      <Plot
        data={plotData}
        layout={layout as PlotlyLayout}
        config={config as PlotlyConfig}
        style={{ width: '100%', height: '100%' }}
        onClick={handlePointClick}
      />
    </div>
  );
};

export default Graph3D;
