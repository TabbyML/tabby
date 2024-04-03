export function getWordStartIndices(text: string): number[] {
  const indices: number[] = [];
  const re = /\b\w/g;
  let match;
  while ((match = re.exec(text)) != null) {
    indices.push(match.index);
  }
  return indices;
}