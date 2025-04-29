// Utility functions for vector operations

// Simple function to calculate distance between two vectors
export const calculateDistance = (vec1: number[], vec2: number[]): number => {
    if (vec1.length !== vec2.length) {
      throw new Error("Vectors must have the same dimensions");
    }
    
    let sum = 0;
    for (let i = 0; i < vec1.length; i++) {
      sum += Math.pow(vec1[i] - vec2[i], 2);
    }
    
    return Math.sqrt(sum);
  };
  
  // Calculate t-SNE like positions (simplified for visualization)
  // In a real app, you'd use a proper t-SNE or UMAP implementation
  export const calculatePositions = (
    vectors: number[][],
    dimensions = 3,
    scale = 5
  ): number[][] => {
    // This is a very simplified approach - in production, use a real dimensionality reduction algorithm
    return vectors.map((vector) => {
      // Just use the first dimensions of the vector for visualization
      // and scale them appropriately
      return vector.slice(0, dimensions).map((v) => v * scale);
    });
  };
  
  // Create edges between nodes that are close to each other
  export const createEdges = (
    positions: number[][],
    threshold = 2.5
  ): [number, number][] => {
    const edges: [number, number][] = [];
    
    for (let i = 0; i < positions.length; i++) {
      for (let j = i + 1; j < positions.length; j++) {
        const distance = calculateDistance(positions[i], positions[j]);
        if (distance < threshold) {
          edges.push([i, j]);
        }
      }
    }
    
    return edges;
  };