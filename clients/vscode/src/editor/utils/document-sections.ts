import * as vscode from 'vscode'

export async function getDocumentSections(
    doc: vscode.TextDocument,
    // Optional overwrites to simplify testing
    getFoldingRanges = defaultGetFoldingRanges,
    getSymbols = defaultGetSymbols
): Promise<vscode.Range[]> {
    // Documents with language ID 'plaintext' do not have symbol support in VS Code
    // In those cases, try to find class ranges heuristically
    const isPlainText = doc.languageId === 'plaintext'

    // Remove imports, comments, and regions from the folding ranges
    const foldingRanges = await getFoldingRanges(doc.uri).then(r => r?.filter(r => !r.kind))
    if (!foldingRanges?.length) {
        return []
    }

    const innerRanges = await removeOutermostFoldingRanges(doc, foldingRanges, getSymbols)

    const ranges = removeNestedFoldingRanges(innerRanges, isPlainText)

    return ranges.map(r => foldingRangeToRange(doc, r))
}

/**
 * Gets the folding range containing the target position.
 * Target position that sits outside of any folding range will return undefined.
 *
 * NOTE: Use getSmartSelection from utils/index.ts instead
 */
export async function getSelectionAroundLine(
    doc: vscode.TextDocument,
    line: number
): Promise<vscode.Selection | undefined> {
    const smartRanges = await getDocumentSections(doc)

    // Filter to only keep folding ranges that contained nested folding ranges (aka removes nested ranges)
    // Get the folding range containing the active cursor
    const range = findRangeByLine(smartRanges, line)

    if (!range) {
        return undefined
    }

    return new vscode.Selection(range.start, range.end)
}

/**
 * Finds the folding range containing the given target position.
 *
 * NOTE: exported for testing purposes only
 *
 * @param ranges - The array of folding ranges to search.
 * @param targetLine - The position to find the containing range for.
 * @returns The folding range containing the target position, or undefined if not found.
 */
export function findRangeByLine(ranges: vscode.Range[], targetLine: number): vscode.Range | undefined {
    return ranges.find(range => range.start.line <= targetLine && range.end.line >= targetLine)
}

const TOO_LARGE_SECTION = 100
/**
 * Gets the outermost folding ranges that are too large to be considered a section. This includes
 * classes in most cases (where we want individual methods to be considered sections), but also
 * works with e.g. jest test files, huge functions and React components.
 */
async function getOutermostFoldingRanges(
    doc: vscode.TextDocument,
    ranges: vscode.FoldingRange[],
    getSymbols: typeof defaultGetSymbols
): Promise<vscode.Range[]> {
    const symbolBased = await getSymbols(doc.uri)
        .then(r =>
            r.filter(
                s =>
                    s.kind === vscode.SymbolKind.Class ||
                    s.kind === vscode.SymbolKind.Module ||
                    s.kind === vscode.SymbolKind.Namespace ||
                    rangeLines(s.location.range) > TOO_LARGE_SECTION
            )
        )
        .then(s => s.map(symbol => symbol.location.range))

    if (symbolBased.length > 0) {
        return symbolBased
    }

    // If the document does not support symbols, we use a heuristics to find the outermost folding
    // ranges

    const outermostFoldingRanges = removeNestedFoldingRanges(ranges)

    // Check outerRanges array for the string 'class' in each starting line to confirm they are
    // class ranges Filter the ranges to remove ranges that did not contain classes in their first
    // line
    const firstLines = outermostFoldingRanges.map(r => doc.lineAt(r.start).text)
    return outermostFoldingRanges
        .map(r => foldingRangeToRange(doc, r))
        .filter(
            (r, i) =>
                firstLines[i].includes('class') ||
                firstLines[i].startsWith('object') ||
                rangeLines(r) > TOO_LARGE_SECTION
        )
}

/**
 * Removes outermost folding ranges from the given folding ranges array.
 */
async function removeOutermostFoldingRanges(
    doc: vscode.TextDocument,
    foldingRanges: vscode.FoldingRange[],
    getSymbols: typeof defaultGetSymbols
): Promise<vscode.FoldingRange[]> {
    const outermostRanges = await getOutermostFoldingRanges(doc, foldingRanges, getSymbols)

    if (!outermostRanges.length || !foldingRanges?.length) {
        return foldingRanges
    }

    for (const oRanges of outermostRanges) {
        for (let i = 0; i < foldingRanges.length; i++) {
            const range = foldingRanges[i]
            if (range.start === oRanges.start.line && Math.abs(range.end - oRanges.end.line) <= 1) {
                foldingRanges.splice(i, 1)
                i--
            }
        }
    }

    return foldingRanges
}

/**
 * Removes nested folding ranges from the given array of folding ranges.
 *
 * This filters the input array to only contain folding ranges that do not have any nested child
 * folding ranges within them.
 *
 * Nested folding ranges occur when you have a folding range (e.g. for a function) that contains
 * additional nested folding ranges (e.g. for inner code blocks).
 *
 * By removing the nested ranges, you are left with only the top-level outermost folding ranges.
 *
 * @param ranges - Array of folding ranges
 * @returns Array containing only folding ranges that do not contain any nested child ranges
 */
function removeNestedFoldingRanges(ranges: vscode.FoldingRange[], isTextBased = false): vscode.FoldingRange[] {
    const filtered = isTextBased ? combineNeighborFoldingRanges(ranges) : ranges

    return filtered.filter(
        cur => !filtered.some(next => next !== cur && next.start <= cur.start && next.end >= cur.end)
    )
}

/**
 * Combines adjacent folding ranges in the given array into single combined ranges.
 *
 * This will iterate through the input ranges, and combine any ranges that are adjacent (end line of previous connects to start line of next)
 * into a single combined range.
 *
 * @param ranges - Array of folding ranges to combine
 * @returns Array of combined folding ranges
 */
function combineNeighborFoldingRanges(ranges: vscode.FoldingRange[]): vscode.FoldingRange[] {
    const combinedRanges: vscode.FoldingRange[] = []

    let currentChain: vscode.FoldingRange[] = []
    let lastChainRange = currentChain.at(-1)

    for (const range of ranges) {
        // set the lastChainRange to the last range in the current chain
        lastChainRange = currentChain.at(-1)
        if (currentChain.length > 0 && lastChainRange?.end === range.start - 1) {
            // If this range connects to the previous one, add it to the current chain
            currentChain.push(range)
        } else {
            // Otherwise, start a new chain
            if (currentChain.length > 0 && lastChainRange) {
                // If there was a previous chain, combine it into a single range
                combinedRanges.push(new vscode.FoldingRange(currentChain[0].start, lastChainRange.end))
            }

            currentChain = [range]
        }
    }

    // Add the last chain
    if (lastChainRange && currentChain.length > 0) {
        combinedRanges.push(new vscode.FoldingRange(currentChain[0].start, lastChainRange.end))
    }

    return combinedRanges
}

const closingSymbols = /^(}|]|\)|>|end|fi|elsif)/

/**
 * Approximates a range that starts at the first character of the first line and ends at the last
 * character of the last line.
 *
 * Note that folding ranges in VS Code do not include the closing brace line. We err on the side of
 * not adding an extra line in cases where the heuristics fails.
 */
function foldingRangeToRange(doc: vscode.TextDocument, range: vscode.FoldingRange): vscode.Range {
    const nextLine = doc.getText(new vscode.Range(range.end + 1, 0, range.end + 2, 0))

    // We include the next line after the folding range starts with a closing symbol
    const includeNextLine = !!nextLine.trim().match(closingSymbols)

    const start = range.start
    const end = range.end + (includeNextLine ? 1 : 0)

    // Get the text of the last line so we can count the chars until the end
    const endLine = doc.getText(new vscode.Range(end, 0, end + 1, 0))

    return new vscode.Range(start, 0, end, endLine.length - 1)
}

function rangeLines(range: vscode.Range): number {
    return range.end.line - range.start.line
}

async function defaultGetSymbols(uri: vscode.Uri): Promise<vscode.SymbolInformation[]> {
    return (
        (await vscode.commands.executeCommand<vscode.SymbolInformation[]>(
            'vscode.executeDocumentSymbolProvider',
            uri
        )) || []
    )
}

async function defaultGetFoldingRanges(uri: vscode.Uri): Promise<vscode.FoldingRange[]> {
    return (
        (await vscode.commands.executeCommand<vscode.FoldingRange[]>('vscode.executeFoldingRangeProvider', uri)) || []
    )
}
