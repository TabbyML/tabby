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
  ListFileItem,
  ListFilesInWorkspaceParams
} from 'tabby-chat-panel/index'

import { cn, resolveFileNameForDisplay } from '@/lib/utils'
import { IconFile } from '@/components/ui/icons'

import type { SourceItem } from './types'
import { fileItemToSourceItem, shortenLabel } from './utils'

/**
 * A React component to render a mention node in the editor.
 * Displays the filename and an icon in a highlighted style.
 */
export const MentionComponent = ({ node }: { node: any }) => {
  return (
    <NodeViewWrapper className="inline-block align-middle -my-1">
      <span
        className={cn(
          'bg-muted prose text-foreground inline-flex gap-0.5 items-center rounded px-1.5 py-0.5 text-sm font-medium',
          'ring-muted ring-1 ring-inset',
          'relative top-[0.1em]'
        )}
        data-category={node.attrs.category}
      >
        <IconFile />
        <span className="relative top-[-0.5px]">
          {resolveFileNameForDisplay(node.attrs.filepath)}
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
    const filePath = node.attrs.filepath
    // let label = ''
    // if (filePath.kind == 'git') {
    //   label = filePath.filepath
    // } else {
    //   label = filePath.uri
    // }
    // If symbols can be mentioned later, the placeholder could be [[symbol:{label}]].
    return `[[file:${filePath}]]`
  },

  // Defines custom attributes for the mention node
  addAttributes() {
    return {
      filepath: {
        default: null,
        parseHTML: element => element.getAttribute('data-filepath'),
        renderHTML: attrs => {
          if (!attrs.id) return {}
          return { 'data-filepath': attrs.filepath }
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
    const onSelectItem = (idx: number) => {
      const item = items[idx]
      if (!item || !command) return
      command({
        category: 'file',
        filepath: item.filepath
      })
    }

    const enterHandler = () => {
      onSelectItem(selectedIndex)
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
      <div className="max-h-[300px] overflow-auto p-1 bg-popover border rounded-md">
        {/* If no items are found, show a message. */}
        {!items.length ? (
          <div className="px-2 py-1.5 text-sm text-muted-foreground">
            Cannot find any files.
          </div>
        ) : (
          <div className="grid gap-0.5">
            {items.map((item, index) => {
              const filepath = item.fileItem.filepath
              return (
                <OptionItemView
                  key={`${JSON.stringify(filepath)}`}
                  onClick={() => onSelectItem(index)}
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
        'flex cursor-pointer gap-1 rounded-md px-2 py-1.5 text-sm items-center flex-nowrap',
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
