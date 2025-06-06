// Sample embedding data for visualization
// In a real application, these would be actual embeddings from an ML model
import jsonData from '../data/res.json';

export interface TextItem {
    id: string;
    title: string;
    author: string;
    year: number;
    type: "novel" | "article" | "poem" | "short story";
    description: string;
    embedding: number[];
    embedding_dim3: number[];
  }
  
  // Load data from JSON file and assert its type
  export const sampleData = jsonData as TextItem[];
  
  export const getUniqueTypes = () => {
    return Array.from(new Set(sampleData.map(item => item.type)));
  };
  
