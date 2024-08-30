import './mention-list.css'

import React, {
  forwardRef,
  useContext,
  useEffect,
  useImperativeHandle,
  useMemo,
  useState
} from 'react'
// import { MentionNodeAttrs } from '@tiptap/extension-mention'
import {
  SuggestionKeyDownProps,
  SuggestionProps
} from '@tiptap/extension-mention/dist/packages/suggestion/src/index.d'
import { go as fuzzy } from 'fuzzysort'

import { cn } from '@/lib/utils'

import { MentionContext } from '.'
import { IconFileText, IconGitFork, IconSpinner } from '../ui/icons'
import { CategoryOptionItem, OptionItem } from './types'
import { getMentionsWithIndices } from './utils'

interface MetionListProps extends SuggestionProps {}

interface MentionListActions {
  onKeyDown: (props: SuggestionKeyDownProps) => void
}

const CATEGORY_OPTIONS: CategoryOptionItem[] = [
  {
    type: 'category',
    label: 'Repository',
    kind: 'code'
  },
  {
    type: 'category',
    label: 'Document',
    kind: 'doc'
  }
]

const MetionList = forwardRef<MentionListActions, MetionListProps>(
  ({ query, command, editor, items }, ref) => {
    const { list, pending } = useContext(MentionContext)

    const json = editor.getJSON()
    const hasSelectedRepo = useMemo(() => {
      // FIXME walk tree?
      const mentions = getMentionsWithIndices(editor)
      return (
        mentions?.findIndex(o => o?.id === 'tabbyCode' || o?.id === 'react') !==
        -1
      )
    }, [json])

    const [selectedIndex, setSelectedIndex] = useState(0)
    const [kind, setKind] = useState<'doc' | 'code' | undefined>()

    const options: OptionItem[] = useMemo(() => {
      if (!query && !kind) {
        return hasSelectedRepo
          ? CATEGORY_OPTIONS.filter(o => o.kind !== 'code')
          : CATEGORY_OPTIONS
      }
      if (!kind || !list) {
        return list ?? []
      }

      const docSources = list.filter(o => o.kind === 'doc')
      const codeSources = list.filter(o => o.kind === 'code')
      return kind === 'doc' ? docSources : hasSelectedRepo ? [] : codeSources
    }, [kind, list, query, hasSelectedRepo])

    const filteredList = useMemo(() => {
      if (!query) return options

      const result = fuzzy(query, options, {
        key: item => item.label
      })
      return result.map(o => o.obj)
    }, [query, options])

    const upHandler = () => {
      setSelectedIndex((selectedIndex + options.length - 1) % options.length)
    }

    const downHandler = () => {
      setSelectedIndex((selectedIndex + 1) % options.length)
    }

    const onSelectItem = (idx: number) => {
      const item = filteredList[idx]
      if (!item) return
      if (item.type === 'category') {
        setKind(item.kind)
      } else {
        command({ id: item.id, label: item.label })
      }
    }

    const enterHandler = () => {
      onSelectItem(selectedIndex)
    }

    useEffect(() => setSelectedIndex(0), [options])

    useImperativeHandle(ref, () => ({
      onKeyDown: ({ event }: { event: KeyboardEvent }) => {
        if (event.key === 'ArrowUp') {
          upHandler()
          return true
        }

        if (event.key === 'ArrowDown') {
          downHandler()
          return true
        }

        if (event.key === 'Enter') {
          enterHandler()
          return true
        }

        return false
      }
    }))

    return (
      <div className="dropdown-menu min-w-[12rem] overflow-hidden rounded-md border bg-popover p-2 text-popover-foreground shadow animate-in">
        {filteredList.length ? (
          filteredList.map((item, index) => (
            <div
              className={cn(
                'cursor-pointer flex items-center gap-1 rounded-md px-2 py-1.5 text-sm',
                {
                  'bg-accent text-accent-foreground': index === selectedIndex
                }
              )}
              key={index}
              onClick={() => onSelectItem(index)}
            >
              <span className="shrink-0">
                {item.kind === 'code' ? <IconGitFork /> : <IconFileText />}
              </span>
              {item.label}
            </div>
          ))
        ) : pending ? (
          <div className="px-2 py-1.5">
            <IconSpinner />
          </div>
        ) : (
          <div className="item">No result</div>
        )}
      </div>
    )
  }
)

MetionList.displayName = 'MetionList'

export default MetionList
