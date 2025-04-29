import { useState } from 'react';
import Graph3D from './Graph3D';
import NodeInfo from './NodeInfo';
import FilterControls, {type FilterOptions } from './FilterControls';
import { sampleData, type TextItem, getUniqueTypes } from '../utils/sampleData';

const Home = () => {
  const [selectedNode, setSelectedNode] = useState<TextItem | null>(null);
  const [filters, setFilters] = useState<FilterOptions>({
    types: getUniqueTypes(),
  });

  return (
    <div className="relative w-full h-screen">
      {/* Beautiful centered title */}
      <div className="absolute top-0 left-0 w-full z-20 pt-6 pb-4 text-center bg-gradient-to-b from-black/90 to-black/0">
        <div className="inline-block px-8 py-4 rounded-lg bg-black/60 backdrop-blur-lg border border-purple-500/30 shadow-lg">
          <h1 className="text-4xl font-bold bg-gradient-to-br from-purple-300 via-pink-300 to-indigo-400 bg-clip-text text-transparent drop-shadow-sm">
            Asian American Studies Literature Visual
          </h1>
          <p className="text-gray-300 text-sm mt-2">
            A digital humanities project exploring relationships between embeddings in 3D vector space
          </p>
        </div>
      </div>

      {/* Graph visualization */}
      <div className="absolute top-0 left-0 w-full h-full z-0">
        <Graph3D 
          data={sampleData} 
          onSelectNode={setSelectedNode}
          filters={filters}
        />
      </div>
      
      {/* Filters panel */}
      <FilterControls onFilterChange={setFilters} />
      
      {/* Selected node info */}
      <NodeInfo item={selectedNode} />
      
      {/* Instructions overlay */}
      <div className="absolute bottom-4 right-4 bg-black/70 p-10 rounded-md text-white text z-10 backdrop-blur-sm border border-white/10">
        <p className="mb-1"><span className="font-bold">Hover on a point</span> to see details</p>
        <p className="mb-1"><span className="font-bold">Drag</span> to rotate view</p>
        <p><span className="font-bold">Scroll</span> to zoom in/out</p>
      </div>
    </div>
  );
};

export default Home;