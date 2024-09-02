import React, {
  forwardRef,
  useContext,
  useEffect,
  useImperativeHandle,
  useMemo,
  useState
} from 'react'
import {
  SuggestionKeyDownProps,
  SuggestionProps
} from '@tiptap/extension-mention/dist/packages/suggestion/src/index.d'
import { go as fuzzy } from 'fuzzysort'

import { ContextKind } from '@/lib/gql/generates/graphql'
import { MentionAttributes } from '@/lib/types'
import { cn, isCodeSourceContext, isDocSourceContext } from '@/lib/utils'

import { MentionContext } from '.'
import {
  IconCode,
  IconFileText,
  IconGitHub,
  IconGitLab,
  IconGlobe,
  IconSpinner
} from '../ui/icons'
import { CategoryOptionItem, OptionItem, SourceOptionItem } from './types'
import { isRepositorySource } from './utils'

export interface MetionListProps extends SuggestionProps {
  mentions?: MentionAttributes[]
}

export interface MentionListActions {
  onKeyDown: (props: SuggestionKeyDownProps) => boolean
}

const CATEGORY_OPTIONS: CategoryOptionItem[] = [
  {
    type: 'category',
    label: 'Repository',
    category: 'code'
  },
  {
    type: 'category',
    label: 'Document',
    category: 'doc'
  }
]

const MetionList = forwardRef<MentionListActions, MetionListProps>(
  ({ query, command, mentions }, ref) => {
    const { list, pending } = useContext(MentionContext)

    const hasSelectedRepo = useMemo(() => {
      return (
        mentions?.findIndex(o => {
          return isRepositorySource(o.kind)
        }) !== -1
      )
    }, [mentions])

    const [selectedIndex, setSelectedIndex] = useState(0)
    const [category, setCategory] = useState<'doc' | 'code' | undefined>()

    const options: OptionItem[] = useMemo(() => {
      if (!query && !category) {
        return hasSelectedRepo
          ? CATEGORY_OPTIONS.filter(o => o.category !== 'code')
          : CATEGORY_OPTIONS
      }
      if (!list?.length) {
        return []
      }

      const docSources: SourceOptionItem[] = list
        .filter(o => isDocSourceContext(o.kind))
        .map(item => ({
          type: 'source',
          category: 'doc',
          id: item.sourceId,
          label: item.displayName,
          data: item
        }))

      const codeSources: SourceOptionItem[] = list
        .filter(o => isCodeSourceContext(o.kind))
        .map(item => ({
          type: 'source',
          category: 'code',
          id: item.sourceId,
          label: item.displayName,
          data: item
        }))

      if (!category) {
        return hasSelectedRepo ? docSources : [...docSources, ...codeSources]
      }

      return category === 'doc'
        ? docSources
        : hasSelectedRepo
        ? []
        : codeSources
    }, [category, list, query, hasSelectedRepo])

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
        setCategory(item.category)
      } else {
        command({
          id: item.data.sourceId,
          label: item.label,
          kind: item.data.kind
        })
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
      <div className="dropdown-menu min-w-[20rem] overflow-x-hidden overflow-y-auto max-h-[30vh] rounded-md border bg-popover p-2 text-popover-foreground shadow animate-in">
        {filteredList.length ? (
          filteredList.map((item, index) => (
            <div
              className={cn(
                'flex cursor-pointer gap-1 rounded-md px-2 py-1.5 text-sm',
                {
                  'bg-accent text-accent-foreground': index === selectedIndex
                }
              )}
              key={index}
              onClick={() => onSelectItem(index)}
              onMouseEnter={() => setSelectedIndex(index)}
              title={item.label}
            >
              <span className="flex h-5 shrink-0 items-center">
                <OptionIcon option={item} />
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
              <span>No results, please configure in Context Providers</span>
            )}
          </div>
        )}
      </div>
    )
  }
)

MetionList.displayName = 'MetionList'

function OptionIcon({ option }: { option: OptionItem }) {
  if (option.type === 'category') {
    return option.category === 'code' ? <IconCode /> : <IconFileText />
  }

  if (option.type === 'source') {
    switch (option.data.kind) {
      case ContextKind.Doc:
        return <IconFileText />
      case ContextKind.Web:
        return <IconGlobe />
      case ContextKind.Git:
        return <IconCode />
      case ContextKind.Github:
        return <IconGitHub />
      case ContextKind.Gitlab:
        return <IconGitLab />
      default:
        return null
    }
  }
}

export default MetionList
