import { LookupDefinitionsHint, SymbolInfo } from "tabby-chat-panel/index";
import {
  chatPanelLocationToVSCodeRange,
  getActualChatPanelFilepath,
  vscodeRangeToChatPanelPositionRange,
} from "./utils";
import { Range as VSCodeRange } from "vscode";

/**
 * Filters out SymbolInfos whose target is inside the given context range,
 * and merges overlapping target ranges in the same file.
 */
export function filterSymbolInfosByContextAndOverlap(
  symbolInfos: SymbolInfo[],
  context: LookupDefinitionsHint | undefined,
): SymbolInfo[] {
  if (!symbolInfos.length) {
    return [];
  }

  // Filter out target inside context
  let filtered = symbolInfos;
  if (context?.location) {
    const contextRange = chatPanelLocationToVSCodeRange(context.location);
    const contextPath = context.filepath ? getActualChatPanelFilepath(context.filepath) : undefined;
    if (contextRange && contextPath) {
      filtered = filtered.filter((symbolInfo) => {
        const targetPath = getActualChatPanelFilepath(symbolInfo.target.filepath);
        if (targetPath !== contextPath) {
          return true;
        }
        // Check if target is outside contextRange
        const targetRange = chatPanelLocationToVSCodeRange(symbolInfo.target.location);
        if (!targetRange) {
          return true;
        }
        return targetRange.end.isBefore(contextRange.start) || targetRange.start.isAfter(contextRange.end);
      });
    }
  }

  // Merge overlapping target ranges in same file
  const merged: SymbolInfo[] = [];
  for (const current of filtered) {
    const currentUri = getActualChatPanelFilepath(current.target.filepath);
    const currentRange = chatPanelLocationToVSCodeRange(current.target.location);
    if (!currentRange) {
      merged.push(current);
      continue;
    }

    // Try find a previously added symbol that is in the same file and has overlap
    let hasMerged = false;
    for (const existing of merged) {
      const existingUri = getActualChatPanelFilepath(existing.target.filepath);
      if (existingUri !== currentUri) {
        continue;
      }
      const existingRange = chatPanelLocationToVSCodeRange(existing.target.location);
      if (!existingRange) {
        continue;
      }
      // Check overlap
      const isOverlap = !(
        currentRange.end.isBefore(existingRange.start) || currentRange.start.isAfter(existingRange.end)
      );
      if (isOverlap) {
        // Merge
        const newStart = currentRange.start.isBefore(existingRange.start) ? currentRange.start : existingRange.start;
        const newEnd = currentRange.end.isAfter(existingRange.end) ? currentRange.end : existingRange.end;
        const mergedRange = new VSCodeRange(newStart, newEnd);
        existing.target.location = vscodeRangeToChatPanelPositionRange(mergedRange);
        hasMerged = true;
        break;
      }
    }
    if (!hasMerged) {
      merged.push(current);
    }
  }
  return merged;
}
