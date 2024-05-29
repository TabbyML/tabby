// Inspired by
// https://github.com/sourcegraph/opencodegraph

import { Facet, type EditorState, type Extension } from '@codemirror/state'
import {
  Decoration,
  EditorView,
  ViewPlugin,
  ViewUpdate,
  WidgetType,
  type DecorationSet
} from '@codemirror/view'
import { groupBy, isEqual } from 'lodash-es'

import type { TCodeTag } from '@/lib/types'

interface CodeTagTooltipDecorationsConfig<T = TCodeTag> {
  createDecoration: (
    state: EditorState,
    container: HTMLElement,
    spec: {
      indent: string | undefined
      items: T[]
    }
  ) => { destroy?: () => void }
}

interface Annotation extends TCodeTag {}

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

class BlockWidget<T = TCodeTag> extends WidgetType {
  private container: HTMLElement | null = null
  private decoration:
    | ReturnType<CodeTagTooltipDecorationsConfig['createDecoration']>
    | undefined

  constructor(
    private readonly state: EditorState,
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
      this.decoration = this.config.createDecoration(
        this.state,
        this.container,
        {
          indent: this.indent,
          items: this.items
        }
      )
    }
    return this.container
  }

  public destroy(): void {
    this.container?.remove()
  }
}

function computeDecorations(
  state: EditorState,
  annotations: Annotation[],
  config: CodeTagTooltipDecorationsConfig
): DecorationSet {
  const temp: Array<{ line: number; annotation: Annotation }> = []
  for (const ann of annotations) {
    const range = ann.name_range
    let lineNumber = -1
    try {
      lineNumber = state.doc.lineAt(range.start)?.number ?? -1
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('line parse error')
    }
    if (lineNumber > -1) {
      temp.push({ line: lineNumber, annotation: ann })
    }
  }

  const annotationMapByLine = groupBy(temp, 'line')
  const lineNumbers = Object.keys(annotationMapByLine)

  let widgets = []
  for (const lineNumber of lineNumbers) {
    const lineItems = annotationMapByLine[lineNumber]?.map(o => o.annotation)
    const line = state.doc.line(Number(lineNumber))
    const indent = line.text.match(/^\s*/)?.[0]
    widgets.push(
      Decoration.widget({
        widget: new BlockWidget(state, lineItems, indent, config),
        side: -1
      }).range(line.from)
    )
  }

  return Decoration.set(widgets)
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

export function getCodeTagWidgetExtension(
  tags: TCodeTag[],
  config: CodeTagTooltipDecorationsConfig
) {
  return [
    ViewPlugin.fromClass(
      class {
        codeTags: DecorationSet
        constructor(view: EditorView) {
          this.codeTags = computeDecorations(view.state, tags, config)
        }
        update(update: ViewUpdate) {
          if (update.docChanged) {
            this.codeTags = computeDecorations(update.view.state, tags, config)
          }
        }
      },
      {
        decorations: instance => {
          return instance.codeTags
        }
      }
    ),
    inlineWidgetTheme
  ]
}

/**
 * example
 */
// getCodeTagWidgetExtension(tags, {
//   createDecoration(state, container, { items, indent }) {
//     const div = document.createElement('div')
//     const root = createRoot(div)
//     root.render(
//       <div className="mt-1 flex flex-nowrap items-center gap-1">
//         {indent ? <SpaceDisplay spaceLength={indent.length} /> : null}
//         {items.map(item => {
//           const { name_range, syntax_type_name } = item
//           const key = `${syntax_type_name}_${name_range.start}_${name_range.end}`
//           const name = state.doc.slice(
//             item.name_range.start,
//             item.name_range.end
//           )
//           return (
//             <div
//               key={key}
//               className="bg-secondary flex cursor-pointer items-center gap-1 rounded-sm border px-1"
//               onClick={e => {
//                 editor.current?.dispatch({
//                   selection: {
//                     anchor: item.name_range.start,
//                     head: item.name_range.start
//                   }
//                 })
//               }}
//             >
//               <IconSymbolFunction />
//               <span>
//                 {syntax_type_name}: {name}
//               </span>
//             </div>
//           )
//         })}
//       </div>
//     )

//     container.append(div)

//     return {
//       destroy() {
//         div.remove()
//       }
//     }
//   }
// })
