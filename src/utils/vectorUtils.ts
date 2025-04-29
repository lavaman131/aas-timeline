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
