import { Point, SyntaxNode } from "web-tree-sitter";
import { isDefined } from "../utils";

/**
 * Returns a descendant node at the start position and three parent nodes.
 */
export function getNodeAtCursorAndParents(
  node: SyntaxNode,
  startPosition: Point
): readonly [
  { readonly name: "at_cursor"; readonly node: SyntaxNode },
  ...{ name: string; node: SyntaxNode }[]
] {
  const atCursorNode = node.descendantForPosition(startPosition);

  const parent = atCursorNode.parent;
  const parents = [parent, parent?.parent, parent?.parent?.parent]
    .filter(isDefined)
    .map((node) => ({
      name: "parents",
      node,
    }));

  return [
    {
      name: "at_cursor",
      node: atCursorNode,
    },
    ...parents,
  ] as const;
}
