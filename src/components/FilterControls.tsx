import { useState } from "react";
import { getUniqueTypes } from "../utils/sampleData";

interface FilterControlsProps {
  onFilterChange: (filters: FilterOptions) => void;
}

export interface FilterOptions {
  types: string[];
}

const FilterControls = ({ onFilterChange }: FilterControlsProps) => {
  const types = getUniqueTypes();
  
  const [selectedTypes, setSelectedTypes] = useState<string[]>(types);

  const handleTypeChange = (type: string) => {
    let newSelectedTypes;
    if (selectedTypes.includes(type)) {
      newSelectedTypes = selectedTypes.filter(t => t !== type);
    } else {
      newSelectedTypes = [...selectedTypes, type];
    }
    setSelectedTypes(newSelectedTypes);
    onFilterChange({ types: newSelectedTypes });
  };

  return (
    <div className="absolute top-28 left-4 bg-black/70 p-4 rounded-md text-white z-10 max-w-xs backdrop-blur-md border border-purple-500/30 shadow-lg">
      <h3 className="text-lg font-semibold mb-3 bg-gradient-to-r from-purple-200 to-indigo-300 bg-clip-text text-transparent">Filters</h3>
      
      <div className="mb-4">
        <h4 className="text-sm font-medium mb-2 text-purple-200">Type</h4>
        <div className="flex flex-wrap gap-2">
          {types.map(type => (
            <button
              key={type}
              onClick={() => handleTypeChange(type)}
              className={`px-2 py-1 text-xs rounded-full transition-colors ${
                selectedTypes.includes(type)
                  ? "bg-purple-600 text-white border border-purple-400/50 shadow-md"
                  : "bg-gray-800 text-gray-300 border border-gray-700"
              }`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FilterControls;