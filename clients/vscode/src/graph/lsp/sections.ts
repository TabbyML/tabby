import * as vscode from "vscode";

import { getDocumentSections } from "../../editor/utils/document-sections";

import { commonKeywords, identifierPattern } from "./languages";

export interface DocumentSection {
  fuzzyName: string | null;
  location: vscode.Location;
}

/**
 * Creates a top level map of a document's sections based on symbol ranges
 *
 * TODO(philipp-spiess): We need advanced heuristics here so that for very large sections we can
 * divide them into subsections.
 */
export async function getGraphDocumentSections(
  document: vscode.TextDocument
): Promise<DocumentSection[]> {
  const label = "build document symbols map";
  performance.mark(label);
  const ranges = await getDocumentSections(document);

  const sections: DocumentSection[] = [];

  for (const range of ranges) {
    sections.push({
      fuzzyName: extractFuzzyName(document, range),
      location: new vscode.Location(document.uri, range),
    });
  }
  performance.mark(label);
  return sections;
}

function extractFuzzyName(
  document: vscode.TextDocument,
  range: vscode.Range
): string | null {
  const content = document.getText(range);

  for (const match of content.matchAll(identifierPattern)) {
    if (match.index === undefined || commonKeywords.has(match[0])) {
      continue;
    }
    return match[0];
  }
  return null;
}
