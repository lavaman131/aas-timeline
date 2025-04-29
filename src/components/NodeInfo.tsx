import React from 'react';
import type { TextItem } from '../utils/sampleData';

interface NodeInfoProps {
  item: TextItem | null;
}

const NodeInfo: React.FC<NodeInfoProps> = ({ item }) => {
  if (!item) return null;

  // Determine background gradient based on item type
  const getBgGradient = (type: string) => {
    switch(type) {
      case 'novel': return 'from-purple-900/80 to-indigo-900/80';
      case 'article': return 'from-blue-900/80 to-sky-900/80';
      case 'poem': return 'from-pink-900/80 to-rose-900/80';
      default: return 'from-emerald-900/80 to-teal-900/80';
    }
  };

  return (
    <div className={`absolute bottom-4 left-4 bg-gradient-to-br ${getBgGradient(item.type)} p-4 rounded-md text-white max-w-md backdrop-blur-sm border border-white/20 shadow-lg animate-fade-in`}>
      <h2 className="text-xl font-bold mb-1 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent drop-shadow">{item.title}</h2>
      <div className="flex items-center gap-2 mb-2 text-sm">
        <span className="text-purple-300 font-medium">{item.author}, {item.year}</span>
        <span className="px-2 py-0.5 bg-purple-700/60 rounded-full text-xs border border-purple-500/30">
          {item.type}
        </span>
        <span className="px-2 py-0.5 bg-indigo-700/60 rounded-full text-xs border border-indigo-500/30">
          {item.genre}
        </span>
      </div>
      <p className="text-gray-200 text-sm leading-relaxed">{item.description}</p>
    </div>
  );
};

export default NodeInfo;