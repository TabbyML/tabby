import { EditorState, StateEffect } from '@codemirror/state'
import {
  Decoration,
  DecorationSet,
  EditorView,
  ViewPlugin,
  ViewUpdate
} from '@codemirror/view'

import type { TCodeTag } from '@/lib/types'

import { getUTF16NameRange } from './utils'

export const hightlightMark = Decoration.mark({ class: 'cm-range-highlight' })

export const tokenHightlightTheme = EditorView.baseTheme({
  '.cm-range-highlight': { backgroundColor: 'hsl(var(--selection))' }
})

function getHightlights(state: EditorState, tags: TCodeTag[]) {
  let highlightRange: { from: number; to: number } | undefined

  const ranges = state.selection.ranges
  loop: for (const range of ranges) {
    for (const tag of tags) {
      const name_range = getUTF16NameRange(state, tag)
      if (!name_range) continue

      const offset = name_range.start - tag.name_range.start
      if (range.from >= name_range.start && range.to <= name_range.end) {
        highlightRange = {
          from: tag.range.start + offset,
          to: tag.range.end + offset
        }
        break loop
      }
    }
  }
  if (!highlightRange) return Decoration.none

  return Decoration.set([
    hightlightMark.range(highlightRange.from, highlightRange.to)
  ])
}

function getHighlightsFromPos(pos: number, tags: TCodeTag[]) {
  let highlightRange: { from: number; to: number } | undefined

  for (const tag of tags) {
    if (pos >= tag.name_range.start && pos <= tag.name_range.end) {
      highlightRange = { from: tag.range.start, to: tag.range.end }
      break
    }
  }
  if (!highlightRange) return Decoration.none

  return Decoration.set([
    hightlightMark.range(highlightRange.from, highlightRange.to)
  ])
}

const hoverHighlightEffect = StateEffect.define<DecorationSet | null>()

const highlightTagExtension = (tags: TCodeTag[]) => {
  const highlightPlugin = ViewPlugin.fromClass(
    class {
      highlight: DecorationSet
      view: EditorView
      timeout: number
      triggerType: 'cursor' | 'hover' | undefined
      constructor(view: EditorView) {
        this.view = view
        this.highlight = getHightlights(view.state, tags)
        this.timeout = -1
        this.triggerType = 'hover'
        // todo add hover event
        // this.view.dom.addEventListener('mousemove', e => {
        //   this.handleMouseListener(e)
        // })
      }
      update(update: ViewUpdate) {
        if (update.selectionSet) {
          this.triggerType = 'cursor'
          this.highlight = getHightlights(update.view.state, tags)
        } else if (this.triggerType !== 'cursor') {
          for (const tr of update.transactions) {
            for (const e of tr.effects) {
              if (e.is(hoverHighlightEffect) && e.value) {
                this.highlight = e.value
                this.triggerType = 'hover'
              }
            }
          }
        }
      }
      handleMouseListener(e: MouseEvent) {
        if (this.timeout !== -1) {
          clearTimeout(this.timeout)
        }
        if (!this.highlight.size) {
          let timer = setTimeout(() => {
            const pos = this.view.posAtCoords({ x: e.clientX, y: e.clientY })
            if (pos !== null) {
              const decorations = getHighlightsFromPos(pos, tags)
              if (decorations.size) {
                this.triggerType = 'hover'
              } else if (this.triggerType === 'cursor') {
                return
              }

              this.view.dispatch({
                effects: hoverHighlightEffect.of(decorations)
              })
            }
          }, 100)
          this.timeout = timer as unknown as number
        }
      }
      destroy() {
        // this.view.dom.removeEventListener('mousemove', this.handleMouseListener)
      }
    },
    {
      decorations: instance => {
        return instance.highlight
      }
    }
  )
  return [highlightPlugin, tokenHightlightTheme]
}

export { highlightTagExtension }
