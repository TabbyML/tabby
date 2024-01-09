// Inspired by
// https://github.com/sourcegraph/opencodegraph

import {
  Facet,
  RangeSetBuilder,
  type EditorState,
  type Extension
} from '@codemirror/state'
import {
  Decoration,
  EditorView,
  // ViewPlugin,
  // ViewUpdate,
  WidgetType,
  type DecorationSet
} from '@codemirror/view'
import { groupBy, isEqual } from 'lodash-es'

import { TRange } from '@/app/browser/components/source-code-browser'

type CodeTag = string
interface CodeTagTooltipDecorationsConfig<T = CodeTag> {
  createDecoration: (
    container: HTMLElement,
    spec: {
      indent: string | undefined
      items: T[]
    }
  ) => { destroy?: () => void }
}

interface Annotation<T = CodeTag> {
  item: CodeTag
  range: TRange
}

// customTheme for display code tag extension
const inlineWidgetTheme = EditorView.theme({
  // Move line number down to the line with code, not the line with the annotations.
  '.cm-lineNumbers': {
    '& .cm-gutterElement': {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'flex-end',
      justifyContent: 'flex-end'
    }
  }
})

class BlockWidget<T = CodeTag> extends WidgetType {
  private container: HTMLElement | null = null
  private decoration:
    | ReturnType<CodeTagTooltipDecorationsConfig['createDecoration']>
    | undefined

  constructor(
    private readonly items: T[],
    private readonly indent: string | undefined,
    private readonly config: CodeTagTooltipDecorationsConfig<T>
  ) {
    super()
  }

  public eq(other: BlockWidget<T>): boolean {
    return isEqual(this.items, other.items)
  }

  public toDOM(): HTMLElement {
    if (!this.container) {
      this.container = document.createElement('div')
      this.decoration = this.config.createDecoration(this.container, {
        indent: this.indent,
        items: this.items
      })
    }
    return this.container
  }

  public destroy(): void {
    this.container?.remove()
    setTimeout(() => this.decoration?.destroy?.(), 0)
  }
}

function computeDecorations(
  state: EditorState,
  annotations: Annotation[],
  config: CodeTagTooltipDecorationsConfig
): DecorationSet {
  const builder = new RangeSetBuilder<Decoration>()

  const temp: Array<{ line: number; annotation: Annotation }> = []
  for (const ann of annotations) {
    const range = ann.range
    let lineNumber = -1
    try {
      lineNumber = state.doc.lineAt(range.start)?.number ?? -1
    } catch (e) {
      console.log('line parse error')
    }
    if (lineNumber > -1) {
      temp.push({ line: lineNumber, annotation: ann })
    }
  }

  const annotationMapByLine = groupBy(temp, 'line')
  const lineNumbers = Object.keys(annotationMapByLine)

  for (const lineNumber of lineNumbers) {
    const lineItems = annotationMapByLine[lineNumber]?.map(
      o => o.annotation?.item
    )
    const line = state.doc.line(Number(lineNumber))
    const indent = line.text.match(/^\s*/)?.[0]
    builder.add(
      line.from,
      line.from,
      Decoration.widget({
        widget: new BlockWidget(lineItems, indent, config),
        side: -1
      })
    )
  }

  return builder.finish()
}

const codeTagsFacet = Facet.define<Annotation[], Annotation[]>({
  combine(values) {
    return values.flat()
  }
})

export function setCodeTagData(data: Annotation[] | undefined): Extension {
  return data ? codeTagsFacet.of(data) : []
}

export function displayCodeTagWidgets(
  config: CodeTagTooltipDecorationsConfig
): Extension {
  return [
    EditorView.decorations.compute(['doc', codeTagsFacet], state => {
      const data = state.facet(codeTagsFacet)
      return computeDecorations(state, data, config)
    }),
    inlineWidgetTheme
  ]
}
