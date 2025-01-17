import {
  forwardRef,
  HTMLAttributes,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState
} from 'react'
import Mention from '@tiptap/extension-mention'
import { NodeViewWrapper, ReactNodeViewRenderer } from '@tiptap/react'
import { SuggestionKeyDownProps, SuggestionProps } from '@tiptap/suggestion'
import {
  Filepath,
  ListFileItem,
  ListFilesInWorkspaceParams
} from 'tabby-chat-panel/index'

import { cn, convertFilepath, resolveFileNameForDisplay } from '@/lib/utils'
import { IconFile } from '@/components/ui/icons'

import { emitter } from '../event-emitter'
import type { SourceItem } from './types'
import { fileItemToSourceItem, shortenLabel } from './utils'

/**
 * A React component to render a mention node in the editor.
 * Displays the filename and an icon in a highlighted style.
 */
export const MentionComponent = ({ node }: { node: any }) => {
  const fileItem = node.attrs.fileItem
  const filepathString = convertFilepath(fileItem.filepath).filepath

  // FIXME(@jueliang) fine a better way to detect the mention
  useEffect(() => {
    emitter.emit('file_mention_update')

    return () => {
      emitter.emit('file_mention_update')
    }
  }, [])

  return (
    <NodeViewWrapper className="-my-1 inline-block align-middle">
      <span
        className={cn(
          'prose inline-flex items-center gap-0.5 rounded bg-muted px-1.5 py-0.5 text-sm font-medium text-foreground',
          'ring-1 ring-inset ring-muted',
          'relative top-[0.1em]'
        )}
        data-category={node.attrs.category}
      >
        <IconFile />
        <span className="relative top-[-0.5px]">
          {resolveFileNameForDisplay(filepathString)}
        </span>
      </span>
    </NodeViewWrapper>
  )
}

/**
 * A custom TipTap extension to handle file mentions (like @filename).
 * When converted to plain text, it produces a placeholder with file info.
 */
export const PromptFormMentionExtension = Mention.extend({
  // Uses ReactNodeViewRenderer for custom node rendering
  addNodeView() {
    return ReactNodeViewRenderer(MentionComponent)
  },

  // When exported as plain text, use a placeholder format
  renderText({ node }) {
    const fileItem = node.attrs.fileItem
    const filePath = fileItem.filepath as Filepath
    // If symbols can be mentioned later, the placeholder could be [[symbol:{label}]].
    return `[[file:${JSON.stringify(filePath)}]]`
  },

  // Defines custom attributes for the mention node
  addAttributes() {
    return {
      id: {
        default: null,
        parseHTML: element => element.getAttribute('data-file'),
        renderHTML: attrs => {
          if (!attrs.fileItem) return {}
          return { 'data-id': JSON.stringify(attrs.fileItem.filepath) }
        }
      },
      fileItem: {
        default: null,
        parseHTML: element => element.getAttribute('data-file'),
        renderHTML: attrs => {
          if (!attrs.fileItem) return {}
          return { 'data-file': attrs.fileItem }
        }
      },
      category: {
        default: 'file',
        parseHTML: element => element.getAttribute('data-category'),
        renderHTML: attrs => {
          if (!attrs.category) return {}
          return { 'data-category': attrs.category }
        }
      }
    }
  }
})

export interface MentionListActions {
  onKeyDown: (props: SuggestionKeyDownProps) => boolean
}

export interface MentionListProps extends SuggestionProps {
  items: SourceItem[]
  listFileInWorkspace?: (
    params: ListFilesInWorkspaceParams
  ) => Promise<ListFileItem[]>
  onSelectItem: (item: SourceItem) => void
}

/**
 * A React component for the mention dropdown list.
 * Displays when a user types '@...' and suggestions are fetched.
 */
export const MentionList = forwardRef<MentionListActions, MentionListProps>(
  ({ items: propItems, command, query, listFileInWorkspace }, ref) => {
    const [items, setItems] = useState<SourceItem[]>(propItems)
    const [selectedIndex, setSelectedIndex] = useState(0)

    const upHandler = () => {
      setSelectedIndex((selectedIndex + items.length - 1) % items.length)
    }

    const downHandler = () => {
      setSelectedIndex((selectedIndex + 1) % items.length)
    }

    /**
     * Handle the user selecting an item from the mention list.
     */
    const handleSelectItem = (idx: number) => {
      const item = items[idx]
      if (!item) return
      command({
        category: 'file',
        fileItem: item.fileItem
      })
      // onSelectItem(item)
    }

    const enterHandler = () => {
      handleSelectItem(selectedIndex)
    }

    useEffect(() => setSelectedIndex(0), [items])

    useEffect(() => {
      const fetchOptions = async () => {
        if (!listFileInWorkspace) return []
        const files = await listFileInWorkspace({ query })
        const result = files?.map(fileItemToSourceItem) || []
        setItems(result)
      }
      fetchOptions()
    }, [query])

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
      <div className="max-h-[300px] max-w-[90vw] min-w-[60vw] rounded-md border bg-background p-1 flex flex-col overflow-hidden">
        <div className="text-muted-foreground text-sm p-1 pl-2">Files</div>
        <div className="flex-1 overflow-y-auto">
          {!items.length ? (
            <div className="px-2 py-1.5 text-xs text-muted-foreground">
              {/* If no items are found, show a message. */}
              {query ? 'No results found' : 'Typing to search...'}
            </div>
          ) : (
            <div className="grid gap-0.5">
              {items.map((item, index) => {
                const filepath = item.fileItem.filepath
                return (
                  <OptionItemView
                    key={`${JSON.stringify(filepath)}`}
                    onClick={() => handleSelectItem(index)}
                    onMouseEnter={() => setSelectedIndex(index)}
                    title={item.name}
                    data={item}
                    isSelected={index === selectedIndex}
                  />
                )
              })}
            </div>
          )}
        </div>
      </div>
    )
  }
)
MentionList.displayName = 'MentionList'

interface OptionItemView extends HTMLAttributes<HTMLDivElement> {
  isSelected: boolean
  data: SourceItem
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
        'flex cursor-pointer flex-nowrap items-center gap-1 rounded-md px-2 py-1.5 text-sm',
        {
          'bg-accent text-accent-foreground': isSelected
        }
      )}
      {...rest}
      ref={ref}
    >
      <span className="flex h-5 shrink-0 items-center">
        <IconFile />
      </span>
      <span className="flex-1 truncate">{shortenLabel(data.name)}</span>
      <span className="max-w-[150px] truncate text-xs text-muted-foreground">
        {shortenLabel(data.filepath, 20)}
      </span>
    </div>
  )
}
