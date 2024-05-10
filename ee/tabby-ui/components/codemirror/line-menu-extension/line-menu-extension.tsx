import { RangeSet, StateEffect, StateField } from '@codemirror/state'
import {
  Decoration,
  DecorationSet,
  EditorView,
  gutter,
  gutterLineClass,
  GutterMarker,
  lineNumbers
} from '@codemirror/view'
import { isNil } from 'lodash-es'
import ReactDOM from 'react-dom/client'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { IconMore } from '@/components/ui/icons'

const selectLinesEffect = StateEffect.define<{ pos: number }>()

const selectedLinesState = StateField.define<RangeSet<GutterMarker>>({
  create() {
    return RangeSet.empty
  },
  update(set, transaction) {
    set = set.map(transaction.changes)
    for (let e of transaction.effects) {
      if (e.is(selectLinesEffect)) {
        if (e.value.pos === -1) {
          set = RangeSet.empty
        } else {
          set = RangeSet.empty.update({
            add: [lineMenuMarker.range(e.value.pos)]
          })
        }
      }
    }
    return set
  }
})

const lineMenuMarker = new (class extends GutterMarker {
  toDOM() {
    const dom = document.createElement('div')
    dom.style.textAlign = 'right'
    dom.className = 'breakpoint'
    const root = ReactDOM.createRoot(dom)
    root.render(
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button className="h-5" size="icon" variant="secondary">
            <IconMore />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem className="cursor-pointer">
            Copy line
          </DropdownMenuItem>
          <DropdownMenuItem className="cursor-pointer">
            Copy permalink
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    )
    return dom
  }
})()

const selectableLineNumberTheme = EditorView.theme({
  '.cm-lineNumbers': {
    cursor: 'pointer',
    color: 'var(--line-number-color)',

    '& .cm-gutterElement:hover': {
      textDecoration: 'underline'
    }
  }
})

const selectedLineTheme = EditorView.theme({
  '.cm-selectedLineGutter': {
    backgroundColor: 'hsl(var(--selected-line))'
  },
  '.cm-selectedLine': {
    backgroundColor: 'hsl(var(--selected-line))'
  }
})

function setSelectedLines(view: EditorView, pos: number) {
  const selectedLines = view.state.field(selectedLinesState)
  let hasSelectedLines = false
  selectedLines.between(pos, pos + 1, () => {
    hasSelectedLines = true
  })
  if (!hasSelectedLines) {
    const line = view.state.doc.lineAt(pos)
    view.dispatch({
      effects: [
        selectLinesEffect.of({ pos }),
        lineHighlightEffect.of({ line: line.number, highlight: true })
      ]
    })
  }
}

function clearSelectedLines(view: EditorView) {
  view.dispatch({
    effects: [
      selectLinesEffect.of({ pos: -1 }),
      lineHighlightEffect.of({ highlight: false })
    ]
  })
}

// 定义 StateEffect 用于设置行高亮
const lineHighlightEffect = StateEffect.define<{
  line?: number
  highlight: boolean
}>()
const lineHighlineField = StateField.define<DecorationSet>({
  create() {
    return Decoration.none
  },
  update(highlights, tr) {
    highlights = highlights.map(tr.changes)
    for (let effect of tr.effects) {
      if (effect.is(lineHighlightEffect)) {
        if (effect.value.highlight && !isNil(effect.value.line)) {
          // 添加行高亮装饰
          const deco = Decoration.line({ class: 'cm-selectedLine' })
          const line = tr.state.doc.line(effect.value.line)
          highlights = Decoration.none.update({
            add: [deco.range(line.from)]
          })
        } else {
          // 如果没有指定行号，则移除所有高亮
          highlights = Decoration.none
        }
      }
    }
    return highlights
  },
  provide: field => EditorView.decorations.from(field)
})

const selectedLineGutterMarker = new (class extends GutterMarker {
  elementClass = 'cm-selectedLineGutter'
})()

const selectedLinesGutterHighlighter = gutterLineClass.compute(
  [selectedLinesState],
  state => {
    let marks: any[] = []
    state.field(selectedLinesState).between(0, state.doc.length, (from, to) => {
      console.log('from', from)
      marks.push(selectedLineGutterMarker.range(from))
    })
    return RangeSet.of(marks)
  }
)

const selectLinesGutter = [
  selectedLinesState,
  lineHighlineField,
  gutter({
    class: 'cm-lineMenuGutter',
    markers: v => v.state.field(selectedLinesState),
    initialSpacer: () => lineMenuMarker,
    domEventHandlers: {
      mousedown(view, line) {
        setSelectedLines(view, line.from)
        return true
      }
    }
  }),
  lineNumbers({
    domEventHandlers: {
      mousedown(view, line) {
        // const lineNumber = view.state.doc.lineAt(line.from).number
        setSelectedLines(view, line.from)
        return false
      }
    }
  }),
  selectedLinesGutterHighlighter,
  selectableLineNumberTheme
  // selectedLineTheme,
]

export { selectLinesGutter, setSelectedLines, clearSelectedLines }
