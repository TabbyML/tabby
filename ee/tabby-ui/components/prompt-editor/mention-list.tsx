import React, {
  forwardRef,
  HTMLAttributes,
  useContext,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import {
  SuggestionKeyDownProps,
  SuggestionProps
} from '@tiptap/extension-mention/dist/packages/suggestion/src/index.d'
import { go as fuzzy } from 'fuzzysort'

import { ContextSourceKind } from '@/lib/gql/generates/graphql'
import { MentionAttributes } from '@/lib/types'
import { cn, isCodeSourceContext, isDocSourceContext } from '@/lib/utils'
import {
  IconCode,
  IconEmojiBook,
  IconEmojiGlobe,
  IconGitHub,
  IconGitLab,
  IconSpinner
} from '@/components/ui/icons'

import { MentionContext } from '.'
import { OptionItem, SourceOptionItem } from './types'

export interface MetionListProps extends SuggestionProps {
  mentions?: MentionAttributes[]
  category: 'doc' | 'code'
}

export interface MentionListActions {
  onKeyDown: (props: SuggestionKeyDownProps) => boolean
}

const MetionList = forwardRef<MentionListActions, MetionListProps>(
  ({ query, command, category }, ref) => {
    const { list, pending } = useContext(MentionContext)

    const [selectedIndex, setSelectedIndex] = useState(0)

    const options: OptionItem[] = useMemo(() => {
      if (!list?.length) {
        return []
      }

      const docSources: SourceOptionItem[] = list
        .filter(o => isDocSourceContext(o.sourceKind))
        .map(item => ({
          type: 'source',
          category: 'doc',
          id: item.sourceId,
          label: item.sourceName,
          data: item
        }))

      const codeSources: SourceOptionItem[] = list
        .filter(o => isCodeSourceContext(o.sourceKind))
        .map(item => ({
          type: 'source',
          category: 'code',
          id: item.sourceId,
          label: item.sourceName,
          data: item
        }))

      return category === 'doc' ? docSources : codeSources
    }, [category, list])

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
      command({
        id: item.data.sourceId,
        label: item.label,
        kind: item.data.sourceKind
      })
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
      <div className="dropdown-menu max-h-[30vh] min-w-[20rem] overflow-y-auto overflow-x-hidden rounded-md border bg-popover p-2 text-popover-foreground shadow animate-in">
        {pending ? (
          <div className="px-2 py-1.5">
            <IconSpinner />
          </div>
        ) : filteredList.length ? (
          filteredList.map((item, index) => (
            <OptionItemView
              key={item.id}
              onClick={() => onSelectItem(index)}
              onMouseEnter={() => setSelectedIndex(index)}
              title={item.label}
              data={item}
              isSelected={index === selectedIndex}
            />
          ))
        ) : (
          <div className="px-2 py-1.5 text-sm text-muted-foreground">
            {options?.length ? (
              <span>No matches results</span>
            ) : (
              <span>No results</span>
            )}
          </div>
        )}
      </div>
    )
  }
)

MetionList.displayName = 'MetionList'

function OptionIcon({ kind }: { kind: ContextSourceKind }) {
  switch (kind) {
    case ContextSourceKind.Doc:
      return <IconEmojiBook />
    case ContextSourceKind.Web:
      return <IconEmojiGlobe />
    case ContextSourceKind.Git:
      return <IconCode />
    case ContextSourceKind.Github:
      return <IconGitHub />
    case ContextSourceKind.Gitlab:
      return <IconGitLab />
    default:
      return null
  }
}

interface OptionItemView extends HTMLAttributes<HTMLDivElement> {
  isSelected: boolean
  data: OptionItem
}
function OptionItemView({ isSelected, data, ...rest }: OptionItemView) {
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    if (isSelected && ref.current) {
      ref.current?.scrollIntoView({
        block: 'nearest',
        inline: 'nearest'
      })
    }
  }, [isSelected])

  return (
    <div
      className={cn(
        'flex cursor-pointer gap-1 rounded-md px-2 py-1.5 text-sm',
        {
          'bg-accent text-accent-foreground': isSelected
        }
      )}
      {...rest}
      ref={ref}
    >
      <span className="flex h-5 shrink-0 items-center">
        <OptionIcon kind={data.data.sourceKind} />
      </span>
      <span className="flex-1">{data.label}</span>
    </div>
  )
}

export default MetionList
