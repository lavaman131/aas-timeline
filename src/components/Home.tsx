import { useState } from 'react';
import Graph3D from './Graph3D';
import NodeInfo from './NodeInfo';
import FilterControls, {type FilterOptions } from './FilterControls';
import { sampleData, type TextItem, getUniqueTypes } from '../utils/sampleData';
import statsHtmlContent from '../data/stats.html?raw';
import InfoIcon from '@mui/icons-material/Info';
import CloseIcon from '@mui/icons-material/Close';
import Tooltip from '@mui/material/Tooltip';


const Home = () => {
  const [selectedNode, setSelectedNode] = useState<TextItem | null>(null);
  const [filters, setFilters] = useState<FilterOptions>({
    types: getUniqueTypes(),
  });
  const [isStatsVisible, setIsStatsVisible] = useState(false); // State for sidebar visibility

  const toggleStatsSidebar = () => {
    setIsStatsVisible(!isStatsVisible);
  };

  return (
    // Main container remains relative to position children
    <div className="relative w-full h-screen overflow-hidden bg-black text-white">
      {/* Beautiful centered title */}
      <div className="absolute top-0 left-0 w-full z-20 pt-6 pb-4 text-center bg-gradient-to-b from-black/90 to-black/0 pointer-events-none"> {/* Added pointer-events-none */}
        <div className="inline-block px-8 py-4 rounded-lg bg-black/60 backdrop-blur-lg border border-purple-500/30 shadow-lg">
          <h1 className="text-4xl font-bold bg-gradient-to-br from-purple-300 via-pink-300 to-indigo-400 bg-clip-text text-transparent drop-shadow-sm">
            Asian American Studies Literature Visual
          </h1>
          <p className="text-gray-300 text-sm mt-2">
            A digital humanities project exploring relationships between embeddings in 3D vector space
          </p>
        </div>
      </div>

      {/* Graph visualization (takes full space) */}
      <div className="absolute inset-0 z-0"> {/* Use inset-0 to fill parent */}
        <Graph3D
          data={sampleData}
          onSelectNode={setSelectedNode}
          filters={filters}
        />
      </div>

      {/* Filters panel (Positioned top-left) */}
      <div className="select-none absolute top-28 left-4 z-10">
         <FilterControls onFilterChange={setFilters} />
      </div>


      {/* Selected node info (Positioned top-right) */}
       <div className="absolute top-4 right-4 z-10">
         <NodeInfo item={selectedNode} />
       </div>


      {/* Instructions overlay (Bottom-right) */}
      <div className="select-none absolute bottom-4 right-4 bg-black/70 p-4 rounded-md text-white text-xs z-10 backdrop-blur-sm border border-white/10">
        <p className="mb-1"><span className="font-bold">Hover</span> details | <span className="font-bold">Drag</span> rotate | <span className="font-bold">Scroll</span> zoom</p>
      </div>

      {/* Stats Toggle Button (Bottom-left) */}
      <Tooltip title={isStatsVisible ? "Hide Stats" : "Show Stats"}>
      <button
        onClick={toggleStatsSidebar}
        className="absolute bottom-4 left-4 z-20 p-2 bg-gray-800/80 hover:bg-gray-700/90 rounded-full text-white backdrop-blur-sm border border-white/10 shadow-lg"
      >
        <InfoIcon className="h-6 w-6" />
      </button>
      </Tooltip>

      {/* Overlay for the stats sidebar */}

      {/* Stats Sidebar */}
      <div
        className={`fixed top-0 right-0 h-full w-full max-w-md bg-gray-900/90 backdrop-blur-md shadow-xl z-30 transform transition-transform duration-300 ease-in-out flex flex-col border-l border-gray-700/50
                    ${isStatsVisible ? 'translate-x-0' : 'translate-x-full'}`} // Slide in/out animation
      >
        {/* Sidebar Header */}
        <div className="flex justify-between items-center p-4 border-b border-gray-700 flex-shrink-0">
          <h2 className="text-lg font-semibold text-gray-100">Correlation Stats</h2>
          <button
            onClick={toggleStatsSidebar}
            className="p-1 rounded-md text-gray-400 hover:text-white hover:bg-gray-700"
            title="Close Stats"
          >
            <CloseIcon className="h-6 w-6" />
          </button>
        </div>

        {/* Sidebar Content - Make it scrollable */}
        {/* Apply styles to make the inner HTML content fit and look better */}
        <div className="flex-grow p-4 overflow-y-auto text-xs">
           <div className="stats-html-content text-gray-200" // Add a class for potential CSS overrides
             dangerouslySetInnerHTML={{ __html: statsHtmlContent }}
           />
        </div>
      </div>
    </div>
  );
};

export default Home;