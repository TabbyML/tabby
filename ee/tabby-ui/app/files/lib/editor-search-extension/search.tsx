// Inspired by
// https://github.com/sourcegraph/sourcegraph-public-snapshot

import {
  closeSearchPanel,
  search as codemirrorSearch,
  findNext,
  findPrevious,
  getSearchQuery,
  searchKeymap,
  SearchQuery,
  setSearchQuery
} from '@codemirror/search'
import {
  Compartment,
  SelectionRange,
  StateEffect,
  TransactionSpec,
  type Text as CodeMirrorText
} from '@codemirror/state'
import {
  EditorView,
  KeyBinding,
  keymap,
  Panel,
  runScopeHandlers,
  ViewUpdate
} from '@codemirror/view'
import {
  debounce,
  filter,
  fromValue,
  makeSubject,
  merge,
  pipe,
  subscribe,
  Subscription,
  tap
} from 'wonka'
import type { Subject } from 'wonka'

export type SearchMatches = Map<number, number>

export interface SearchPanelState {
  searchQuery: SearchQuery
  // The input value is usually derived from searchQuery. But we are
  // debouncing updating the searchQuery and without tracking the input value
  // separately user input would be lossing characters and feel laggy.
  inputValue: string
  matches: SearchMatches
  // Currently selected 1-based match index.
  currentMatchIndex: number | null
}

export interface SearchPanelView {
  input: HTMLInputElement | null
  update(state: SearchPanelState): void
  destroy(): void
}

export interface SearchPanelViewCreationOptions {
  root: HTMLElement
  initialState: SearchPanelState
  onSearch: (search: string) => void
  findNext: () => void
  findPrevious: () => void
  setCaseSensitive: (caseSensitive: boolean) => void
  setRegexp: (regexp: boolean) => void
  close: () => void
}

export interface SearchPanelConfig {
  searchValue: string
  regexp: boolean
  caseSensitive: boolean
}

function getMatchIndexForSelection(
  matches: SearchMatches,
  range: SelectionRange
): number | null {
  return range.empty ? null : matches.get(range.from) ?? null
}

// Announce the current match to screen readers.
// Taken from original the CodeMirror implementation:
// https://github.com/codemirror/search/blob/affb772655bab706e08f99bd50a0717bfae795f5/src/search.ts#L694-L717
const announceMargin = 30
const breakRegex = /[\s!,.:;?]/
function announceMatch(
  view: EditorView,
  { from, to }: { from: number; to: number }
): StateEffect<string> {
  const line = view.state.doc.lineAt(from)
  const lineEnd = view.state.doc.lineAt(to).to
  const start = Math.max(line.from, from - announceMargin)
  const end = Math.min(lineEnd, to + announceMargin)
  let text = view.state.sliceDoc(start, end)
  if (start !== line.from) {
    for (let index = 0; index < announceMargin; index++) {
      if (!breakRegex.test(text[index + 1]) && breakRegex.test(text[index])) {
        text = text.slice(index)
        break
      }
    }
  }
  if (end !== lineEnd) {
    for (
      let index = text.length - 1;
      index > text.length - announceMargin;
      index--
    ) {
      if (!breakRegex.test(text[index - 1]) && breakRegex.test(text[index])) {
        text = text.slice(0, index)
        break
      }
    }
  }

  return EditorView.announce.of(
    `${view.state.phrase('current match')}. ${text} ${view.state.phrase(
      'on line'
    )} ${line.number}.`
  )
}

function calculateMatches(
  query: SearchQuery,
  document: CodeMirrorText
): SearchMatches {
  const newSearchMatches: SearchMatches = new Map()

  if (!query.valid) {
    return newSearchMatches
  }

  let index = 1
  const matches = query.getCursor(document)
  let result = matches.next()

  while (!result.done) {
    if (result.value.from !== result.value.to) {
      newSearchMatches.set(result.value.from, index++)
    }

    result = matches.next()
  }

  return newSearchMatches
}

const focusSearchInput = StateEffect.define<boolean>()

class SearchPanel implements Panel {
  public dom: HTMLElement
  public top = true

  private state: SearchPanelState
  private panel: SearchPanelView | null = null
  private searchTerm: Subject<string> = makeSubject<string>()
  private subscriptions: Array<Subscription> = []
  private previousValue: string | null = null

  constructor(
    private view: EditorView,
    private createPanelView: (
      options: SearchPanelViewCreationOptions
    ) => SearchPanelView,
    config?: SearchPanelConfig
  ) {
    this.dom = this.createDom()

    const searchQuery = getSearchQuery(this.view.state)
    const matches = calculateMatches(searchQuery, view.state.doc)
    this.state = {
      searchQuery: new SearchQuery({
        ...searchQuery,
        caseSensitive: config?.caseSensitive ?? searchQuery.caseSensitive,
        regexp: config?.regexp ?? searchQuery.regexp,
        search: config?.searchValue ?? searchQuery.search
      }),
      inputValue: config?.searchValue ?? searchQuery.search,
      matches,
      currentMatchIndex: getMatchIndexForSelection(
        matches,
        view.state.selection.main
      )
    }

    this.subscriptions.push(
      pipe(
        merge([
          fromValue(this.state.searchQuery.search),
          this.searchTerm.source
        ]),
        filter(value => {
          const isDistinct = value !== this.previousValue
          this.previousValue = value
          return isDistinct
        }),
        tap(value => {
          this.state = { ...this.state, inputValue: value }
          this.panel?.update(this.state)
        }),
        debounce(() => 100),
        subscribe(searchTerm => this.commit({ search: searchTerm }))
      )
    )
  }

  private createDom() {
    const dom = document.createElement('div')
    dom.onkeydown = this.onkeydown

    return dom
  }

  public update(update: ViewUpdate): void {
    let newState = this.state

    const searchQuery = getSearchQuery(update.state)
    const searchQueryChanged = !searchQuery.eq(this.state.searchQuery)
    if (searchQueryChanged) {
      newState = {
        ...newState,
        inputValue: searchQuery.search,
        searchQuery,
        matches: calculateMatches(searchQuery, update.view.state.doc)
      }
    }

    // It looks like update.SelectionSet is not set when the search query changes
    if (searchQueryChanged || update.selectionSet) {
      newState = {
        ...newState,
        currentMatchIndex: getMatchIndexForSelection(
          newState.matches,
          update.view.state.selection.main
        )
      }
    }

    if (newState !== this.state) {
      this.state = newState
      this.panel?.update(this.state)
    }

    if (
      update.transactions.some(transaction =>
        transaction.effects.some(
          effect => effect.is(focusSearchInput) && effect.value
        )
      )
    ) {
      this.panel?.input?.focus()
      this.panel?.input?.select()
    }
  }

  public mount(): void {
    this.panel = this.createPanelView({
      root: this.dom,
      initialState: this.state,
      onSearch: search => this.searchTerm.next(search),
      findNext: this.findNext,
      findPrevious: this.findPrevious,
      setCaseSensitive: caseSensitive => this.commit({ caseSensitive }),
      setRegexp: regexp => this.commit({ regexp }),
      close: () => closeSearchPanel(this.view)
    })
  }

  public destroy(): void {
    this.subscriptions.forEach(s => s.unsubscribe())
    this.panel?.destroy()
  }

  private findNext = (): void => {
    findNext(this.view)
    this.view.dispatch({
      effects: EditorView.scrollIntoView(this.view.state.selection.main.from, {
        y: 'nearest',
        yMargin: 20
      })
    })
  }

  private findPrevious = (): void => {
    findPrevious(this.view)
    this.view.dispatch({
      effects: EditorView.scrollIntoView(this.view.state.selection.main.from, {
        y: 'nearest',
        yMargin: 20
      })
    })
  }

  // Taken from CodeMirror's default search panel implementation. This is
  // necessary so that pressing Meta+F (and other CodeMirror keybindings) will
  // trigger the configured event handlers and not just fall back to the
  // browser's default behavior.
  private onkeydown = (event: KeyboardEvent): void => {
    if (runScopeHandlers(this.view, event, 'search-panel')) {
      event.preventDefault()
    } else if (event.key === 'Enter' && event.target === this.panel?.input) {
      event.preventDefault()
      if (event.shiftKey) {
        this.findPrevious()
      } else {
        this.findNext()
      }
    }
  }

  private commit = ({
    search,
    caseSensitive,
    regexp
  }: {
    search?: string
    caseSensitive?: boolean
    regexp?: boolean
  }): void => {
    const query = new SearchQuery({
      search: search ?? this.state.searchQuery.search,
      caseSensitive: caseSensitive ?? this.state.searchQuery.caseSensitive,
      regexp: regexp ?? this.state.searchQuery.regexp
    })

    if (!query.eq(this.state.searchQuery)) {
      let transactionSpec: TransactionSpec = {}
      const effects: StateEffect<any>[] = [setSearchQuery.of(query)]

      if (query.search) {
        // The following code scrolls next match into view if there is no
        // match in the visible viewport. This is done by searching for the
        // text from the currently top visible line and determining whether
        // the next match is in the current viewport

        const { scrollTop } = this.view.scrollDOM

        // Get top visible line. More than half of the line must be visible.
        // We don't use `view.viewportLineBlocks` because that also includes
        // lines that are rendered but not actually visible.
        let topLineBlock = this.view.lineBlockAtHeight(scrollTop)
        if (
          Math.abs(topLineBlock.bottom - scrollTop) <=
          topLineBlock.height / 2
        ) {
          topLineBlock = this.view.lineBlockAtHeight(
            scrollTop + topLineBlock.height
          )
        }

        if (query.regexp && !query.valid) {
          return
        }

        let result = query
          .getCursor(this.view.state.doc, topLineBlock.from)
          .next()
        if (result.done) {
          // No match in the remainder of the document, wrap around
          result = query.getCursor(this.view.state.doc).next()
        }

        if (!result.done) {
          // Taken from the original `findPrevious` and `findNext` CodeMirror implementation:
          // https://github.com/codemirror/search/blob/affb772655bab706e08f99bd50a0717bfae795f5/src/search.ts#L385-L416

          transactionSpec = {
            selection: { anchor: result.value.from, head: result.value.to },
            scrollIntoView: true,
            userEvent: 'select.search'
          }
          effects.push(announceMatch(this.view, result.value))
        }
        // Search term is not in the document, nothing to do
      }

      this.view.dispatch({
        ...transactionSpec,
        effects
      })
    }
  }
}

interface SearchConfig {
  createPanel: (options: SearchPanelViewCreationOptions) => SearchPanelView
  initialState?: SearchPanelConfig
}

export const search = (config: SearchConfig) => {
  const keymapCompartment = new Compartment()

  function getKeyBindings(): readonly KeyBinding[] {
    return searchKeymap.map(binding =>
      binding.key === 'Mod-f'
        ? {
            ...binding,
            run: view => {
              // By default pressing Mod+f when the search input is already focused won't select
              // the input value, unlike browser's built-in search feature.
              // We are overwriting the keybinding here to ensure that the input value is always
              // selected.
              const result = binding.run?.(view)
              if (result) {
                view.dispatch({ effects: focusSearchInput.of(true) })
                return true
              }
              return false
            }
          }
        : binding
    )
  }

  return [
    keymapCompartment.of(keymap.of(getKeyBindings())),
    codemirrorSearch({
      createPanel(view) {
        return new SearchPanel(view, config.createPanel, config.initialState)
      }
    }),
    EditorView.theme({
      '.cm-panels': {
        backgroundColor: 'hsl(var(--secondary))',
        color: 'hsl(var(--secondary-foreground))',
        borderBottom: '1px solid hsl(var(--border))',
        borderTop: '1px solid hsl(var(--border))',
        position: 'sticky',
        top: '50px !important',
        zIndex: 20
      }
    })
  ]
}
