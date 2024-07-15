import {
  RangeSet,
  RangeSetBuilder,
  StateEffect,
  StateField
} from '@codemirror/state'
import {
  Decoration,
  EditorView,
  gutter,
  gutterLineClass,
  GutterMarker,
  lineNumbers
} from '@codemirror/view'
import ReactDOM from 'react-dom/client'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { IconMore } from '@/components/ui/icons'
import { emitter } from '@/app/files/lib/event-emitter'

type SelectedLinesRange = {
  // line number
  line: number
  // line number
  endLine?: number
} | null

const selectedLineGutterMarker = new (class extends GutterMarker {
  elementClass = 'cm-selectedLineGutter'
})()

const selectLinesEffect = StateEffect.define<SelectedLinesRange>()

const selectedLinesField = StateField.define<SelectedLinesRange>({
  create() {
    return null
  },
  update(value, tr) {
    for (const effect of tr.effects) {
      if (effect.is(selectLinesEffect)) {
        return effect.value
      }
      if (effect.is(setEndLine)) {
        if (!value?.line) {
          value = { line: effect.value }
        }
        return { ...value, endLine: effect.value }
      }
    }
    return value
  },
  provide(field) {
    return [
      selectedLinesHighlighter(field),
      selectedLinesGutterHighlighter(field)
    ]
  }
})

const LineMenuButton = ({ isMulti }: { isMulti?: boolean }) => {
  const onCopyLines = () => {
    emitter.emit('line_menu_action', {
      action: 'copy_line'
    })
  }

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Button className="ml-1 h-5" size="icon" variant="secondary">
          <IconMore />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuItem className="cursor-pointer" onSelect={onCopyLines}>
          {isMulti ? 'Copy lines' : 'Copy line'}
        </DropdownMenuItem>
        <DropdownMenuItem
          className="cursor-pointer"
          onSelect={e => {
            emitter.emit('line_menu_action', {
              action: 'copy_permalink'
            })
          }}
        >
          Copy link
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// for single line
const selectedLineMenuButton = new (class extends GutterMarker {
  toDOM(view: EditorView) {
    const range = view.state.field(selectedLinesField)
    const dom = document.createElement('div')
    const root = ReactDOM.createRoot(dom)

    const isMulti = !!range?.endLine

    root.render(<LineMenuButton isMulti={isMulti} />)
    return dom
  }
})()

// for multi lines
const selectedLinesMenuButton = new (class extends GutterMarker {
  toDOM(view: EditorView) {
    const range = view.state.field(selectedLinesField)
    const dom = document.createElement('div')
    const root = ReactDOM.createRoot(dom)

    const isMulti = !!range?.endLine

    root.render(<LineMenuButton isMulti={isMulti} />)
    return dom
  }
})()

function selectedLinesHighlighter(field: StateField<SelectedLinesRange>) {
  return EditorView.decorations.compute([field], state => {
    const range = state.field(field)
    if (!range) {
      return Decoration.none
    }
    const endLine = range.endLine ?? range.line
    const from = Math.min(range.line, endLine)
    const to = Math.min(
      state.doc.lines,
      from === endLine ? range.line : endLine
    )

    const builder = new RangeSetBuilder<Decoration>()

    for (let lineNumber = from; lineNumber <= to; lineNumber++) {
      const from = state.doc.line(lineNumber).from
      builder.add(from, from, Decoration.line({ class: 'cm-selectedLine' }))
    }

    return builder.finish()
  })
}

function selectedLinesGutterHighlighter(field: StateField<SelectedLinesRange>) {
  return gutterLineClass.compute([field], state => {
    let marks: any[] = []
    const range = state.field(field)
    if (!range) {
      return RangeSet.empty
    }
    const endLine = range.endLine ?? range.line
    const from = Math.min(range.line, endLine)
    const to = Math.min(
      state.doc.lines,
      from === endLine ? range.line : endLine
    )
    for (let lineNumber = from; lineNumber <= to; lineNumber++) {
      const from = state.doc.line(lineNumber).from
      marks.push(selectedLineGutterMarker.range(from))
    }
    return RangeSet.of(marks)
  })
}

const selectableLineNumberTheme = EditorView.theme({
  '.cm-lineMenuGutter': {
    width: '40px'
  },
  '.cm-lineNumbers': {
    cursor: 'pointer',
    color: 'var(--line-number-color)',

    '& .cm-gutterElement:hover': {
      textDecoration: 'underline'
    }
  }
})

function setSelectedLines(
  view: EditorView,
  newRange: SelectedLinesRange
): SelectedLinesRange {
  // FIXME check if the range is valid
  view.dispatch({
    effects: selectLinesEffect.of(newRange)
  })
  return newRange
}

function clearSelectedLines(view: EditorView) {
  view.dispatch({
    effects: [selectLinesEffect.of({ line: -1 })]
  })
}

const setEndLine = StateEffect.define<number>()

type SelectLInesGutterOptions = {
  onSelectLine?: (range: SelectedLinesRange | undefined) => void
}

function getMarkersFromSelectedLinesField(view: EditorView) {
  const range = view.state.field(selectedLinesField)
  // FIXME compare line and endLine from range
  // FIXME check if range is valid
  if (range?.line) {
    const pos = view.state.doc.line(range.line).from
    const isMulti = !!range.endLine
    return RangeSet.empty.update({
      add: [
        isMulti
          ? selectedLinesMenuButton.range(pos)
          : selectedLineMenuButton.range(pos)
      ]
    })
  }
  return RangeSet.empty
}

const selectLinesGutter = ({ onSelectLine }: SelectLInesGutterOptions) => {
  return [
    selectableLineNumberTheme,
    selectedLinesField,
    gutter({
      class: 'cm-lineMenuGutter',
      markers: v => getMarkersFromSelectedLinesField(v),
      initialSpacer: () => selectedLineMenuButton,
      domEventHandlers: {
        mousedown(view, line, event) {
          const mouseEvent = event as MouseEvent
          const lineInfo = view.state.doc.lineAt(line.from)
          const lineNumber = lineInfo.number
          view.dispatch({
            effects: mouseEvent.shiftKey
              ? setEndLine.of(lineNumber)
              : selectLinesEffect.of({ line: lineNumber })
          })
          return true
        }
      }
    }),
    lineNumbers({
      domEventHandlers: {
        mousedown(view, line, event) {
          const mouseEvent = event as MouseEvent
          const lineNumber = view.state.doc.lineAt(line.from).number
          view.dispatch({
            effects: mouseEvent.shiftKey
              ? setEndLine.of(lineNumber)
              : selectLinesEffect.of({ line: lineNumber })
          })
          onSelectLine?.(
            formatSelectedLinesRange(view.state.field(selectedLinesField))
          )
          return false
        }
      }
    })
  ]
}

function formatSelectedLinesRange(
  range: SelectedLinesRange
): SelectedLinesRange | undefined {
  if (!range) return undefined

  const { line, endLine } = range

  if (line && endLine) {
    return line === endLine
      ? { line }
      : {
          line: Math.min(line, endLine),
          endLine: Math.max(line, endLine)
        }
  } else if (line) {
    return { line }
  }
}

export {
  selectLinesGutter,
  setSelectedLines,
  clearSelectedLines,
  formatSelectedLinesRange
}
