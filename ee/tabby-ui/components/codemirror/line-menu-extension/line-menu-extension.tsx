import { Button } from '@/components/ui/button';
import { IconMore } from '@/components/ui/icons';
import { RangeSet, StateEffect, StateField } from '@codemirror/state'
import { EditorView, gutter, GutterMarker } from '@codemirror/view'
import ReactDOM from 'react-dom/client'

// const emptyMarker = new class extends GutterMarker {
//   toDOM() { return document.createTextNode("ø") }
// }

// const emptyLineGutter = gutter({
//   lineMarker(view, line) {
//     return line.from == line.to ? emptyMarker : null
//   },
//   initialSpacer: () => emptyMarker
// })

const breakpointEffect = StateEffect.define<{ pos: number; on: boolean }>({
  map: (val, mapping) => ({ pos: mapping.mapPos(val.pos), on: val.on })
})

const breakpointState = StateField.define<RangeSet<GutterMarker>>({
  create() {
    return RangeSet.empty
  },
  update(set, transaction) {
    set = set.map(transaction.changes)
    for (let e of transaction.effects) {
      if (e.is(breakpointEffect)) {
        if (e.value.on)
          set = set.update({ add: [lineMenuMarker.range(e.value.pos)] })
        else set = set.update({ filter: from => from != e.value.pos })
      }
    }
    return set
  }
})

const lineMenuMarker = new (class extends GutterMarker {
  toDOM() {
    const dom = document.createElement('div')
    dom.style.textAlign = 'right'
    const root = ReactDOM.createRoot(dom)
    root.render((
      <Button className='h-5' size='icon'>
        <IconMore />
      </Button>
    ))
    return dom
    // return document.createTextNode('💔')
  }
})()

const breakpointGutter = [
  breakpointState,
  gutter({
    class: 'cm-lineMenuGutter',
    markers: v => v.state.field(breakpointState),
    initialSpacer: () => lineMenuMarker,
    domEventHandlers: {
      mousedown(view, line) {
        toggleBreakpoint(view, line.from)
        return true
      }
    }
  }),
  EditorView.baseTheme({
    '.cm-lineMenuGutter .cm-gutterElement': {
      color: 'red',
      paddingLeft: '5px',
      cursor: 'default'
    }
  })
]

function toggleBreakpoint(view: EditorView, pos: number) {
  let breakpoints = view.state.field(breakpointState)
  let hasBreakpoint = false
  breakpoints.between(pos, pos, () => {
    hasBreakpoint = true
  })
  view.dispatch({
    effects: breakpointEffect.of({ pos, on: !hasBreakpoint })
  })
}

export { breakpointGutter }
