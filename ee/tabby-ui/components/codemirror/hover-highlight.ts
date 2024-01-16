// import { Extension, StateEffect, StateField } from '@codemirror/state'
// import { Decoration, DecorationSet, EditorView } from '@codemirror/view'
// import { isEmpty, isNil } from 'lodash-es'

// import { TCodeTag } from '@/app/browser/components/source-code-browser'

// import { hightlightMark } from './tag-range-highlight-extension'

// const highline2Theme = EditorView.baseTheme({
//   '.cm-underline': { textDecoration: 'underline 3px red' }
// })

// const addHighline2 = StateEffect.define<{ from: number; to: number } | null>({
//   map: (value, change) => {
//     if (!value) {
//       return null
//     } else {
//       const { from, to } = value
//       return {
//         from: change.mapPos(from),
//         to: change.mapPos(to)
//       }
//     }
//   }
// })

// const underline2Field = StateField.define<DecorationSet>({
//   create() {
//     return Decoration.none
//   },
//   update(underlines, tr) {
//     underlines = underlines.map(tr.changes)
//     for (let e of tr.effects)
//       if (e.is(addHighline2)) {
//         if (e.value) {
//           underlines = underlines.update({
//             add: [hightlightMark.range(e.value.from, e.value.to)]
//           })
//         } else {
//           underlines = underlines.update({
//             filter: () => false
//           })
//         }
//       }
//     return underlines
//   },
//   provide: f => EditorView.decorations.from(f)
// })

// export function underlineSelection(view: EditorView, from: number, to: number) {
//   if (!isNil(from) && !isNil(to)) {
//     let effects: StateEffect<unknown>[] = [addHighline2.of({ from, to })]

//     if (!view.state.field(underline2Field, false))
//       effects.push(
//         StateEffect.appendConfig.of([underline2Field, highline2Theme])
//       )
//     view.dispatch({ effects })
//   }
// }

// export function remvoeUnderlineSelection(view: EditorView) {
//   view.dispatch({ effects: addHighline2.of(null) })
// }

// const updateDecorationsEffect = StateEffect.define<DecorationSet>()

// function getHightlights(pos: number, tags: TCodeTag[]) {
//   let highlightRange: { from: number; to: number } | undefined

//   for (const tag of tags) {
//     if (pos >= tag.name_range.start && pos <= tag.name_range.end) {
//       highlightRange = { from: tag.range.start, to: tag.range.end }
//       break
//     }
//   }

//   if (!highlightRange) return []

//   return [highlightRange.from, highlightRange.to]
// }

// function otherEffectsExtension(tags: TCodeTag[]): Extension {
//   let timeout = -1
//   return EditorView.domEventHandlers({
//     mousemove(e, view) {
//       clearTimeout(timeout)
//       timeout = setTimeout(() => {
//         const pos = view.posAtCoords({ x: e.clientX, y: e.clientY })
//         if (pos !== null) {
//           const decorations = getHightlights(pos, tags)
//           if (!isEmpty(decorations)) {
//             underlineSelection(view, decorations?.[0], decorations?.[1])
//           } else {
//             remvoeUnderlineSelection(view)
//           }
//         }
//       })
//       return false
//     }
//   })
// }

// export { otherEffectsExtension }
