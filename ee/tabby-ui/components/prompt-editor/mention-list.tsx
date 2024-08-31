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

import { ContextKind } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'

import { MentionContext } from '.'
import { IconCode, IconFileText, IconSpinner } from '../ui/icons'
import {
  CategoryOptionItem,
  MentionDataItem,
  OptionItem,
  SourceOptionItem
} from './types'
import {
  generateMentionId,
  getInfoFromMentionId,
  isRepositorySource
} from './utils'

export interface MetionListProps extends SuggestionProps {
  mentions?: MentionDataItem[]
}

export interface MentionListActions {
  onKeyDown: (props: SuggestionKeyDownProps) => boolean
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
  ({ query, command, mentions }, ref) => {
    const { list, pending } = useContext(MentionContext)

    const hasSelectedRepo = useMemo(() => {
      return (
        mentions?.findIndex(o => {
          const { kind } = getInfoFromMentionId(o.id)
          return isRepositorySource(kind)
        }) !== -1
      )
    }, [mentions])

    const [selectedIndex, setSelectedIndex] = useState(0)
    const [kind, setKind] = useState<'doc' | 'code' | undefined>()

    const options: OptionItem[] = useMemo(() => {
      if (!query && !kind) {
        return hasSelectedRepo
          ? CATEGORY_OPTIONS.filter(o => o.kind !== 'code')
          : CATEGORY_OPTIONS
      }
      if (!list?.length) {
        return []
      }

      const docSources: SourceOptionItem[] = list
        .filter(o => o.kind === ContextKind.Doc)
        .map(item => ({
          type: 'source',
          kind: 'doc',
          id: generateMentionId(item),
          label: item.displayName,
          data: item
        }))
      const codeSources: SourceOptionItem[] = list
        .filter(o => isRepositorySource(o.kind))
        .map(item => ({
          type: 'source',
          kind: 'code',
          id: generateMentionId(item),
          label: item.displayName,
          data: item
        }))

      if (!kind) {
        return hasSelectedRepo ? docSources : [...docSources, ...codeSources]
      }

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
      setSelectedIndex(
        (selectedIndex + filteredList.length - 1) % filteredList.length
      )
    }

    const downHandler = () => {
      setSelectedIndex((selectedIndex + 1) % filteredList.length)
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
      <div className="dropdown-menu min-w-[20rem] overflow-hidden rounded-md border bg-popover p-2 text-popover-foreground shadow animate-in">
        {filteredList.length ? (
          filteredList.map((item, index) => (
            <div
              className={cn(
                'cursor-pointer flex gap-1 rounded-md px-2 py-1.5 text-sm',
                {
                  'bg-accent text-accent-foreground': index === selectedIndex
                }
              )}
              key={index}
              onClick={() => onSelectItem(index)}
              onMouseEnter={() => setSelectedIndex(index)}
              title={item.label}
            >
              <span className="shrink-0 flex h-5 items-center">
                {item.kind === 'code' ? <IconCode /> : <IconFileText />}
              </span>
              <span>{item.label}</span>
            </div>
          ))
        ) : pending ? (
          <div className="px-2 py-1.5">
            <IconSpinner />
          </div>
        ) : (
          <div className="px-2 py-1.5">
            {list?.length ? (
              <span>No matches results</span>
            ) : (
              <span>No results, please configure in 'Context Providers'</span>
            )}
          </div>
        )}
      </div>
    )
  }
)

MetionList.displayName = 'MetionList'

export default MetionList
