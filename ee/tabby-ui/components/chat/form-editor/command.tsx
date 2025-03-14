import React, {
  forwardRef,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import Mention from '@tiptap/extension-mention'
import { NodeViewWrapper, ReactNodeViewRenderer } from '@tiptap/react'
import { SuggestionKeyDownProps, SuggestionProps } from '@tiptap/suggestion'
import { Loader2 } from 'lucide-react'
import { ChatCommand } from 'tabby-chat-panel'

import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { cn } from '@/lib/utils'

/**
 * A React component to render a command node in the editor.
 * Displays the command name and an icon in a highlighted style.
 */
export const CommandComponent = ({ node }: { node: any }) => {
  const { label } = node.attrs

  return (
    <NodeViewWrapper as="span" className="rounded-sm px-1">
      <span
        className={cn(
          'space-x-0.5 whitespace-nowrap rounded bg-primary/10 px-1.5 py-0.5 align-middle text-sm font-medium text-primary'
        )}
        data-category="command"
      >
        <span className="relative whitespace-normal">/{label}</span>
      </span>
    </NodeViewWrapper>
  )
}

/**
 * A custom TipTap extension to handle slash commands (like /command).
 * We extend Mention but set a unique name to avoid key conflicts.
 */
export const PromptFormCommandExtension = Mention.extend({
  // Set a unique name for this extension to avoid key conflicts with the mention extension
  name: 'slashCommand',

  // Uses ReactNodeViewRenderer for custom node rendering
  addNodeView() {
    return ReactNodeViewRenderer(CommandComponent)
  },

  renderText({ node }) {
    return `[[command:${node.attrs.label}]]`
  },
  addAttributes() {
    return {
      id: {
        default: null,
        parseHTML: element => element.getAttribute('data-command-id'),
        renderHTML: attrs => {
          if (!attrs.commandId) return {}
          return { 'data-command-id': attrs.commandId }
        }
      },
      commandId: {
        default: null,
        parseHTML: element => element.getAttribute('data-command-id'),
        renderHTML: attrs => {
          if (!attrs.commandId) return {}
          return { 'data-command-id': attrs.commandId }
        }
      },
      label: {
        default: '',
        parseHTML: element => element.getAttribute('data-label'),
        renderHTML: attrs => {
          if (!attrs.label) return {}
          return { 'data-label': attrs.label }
        }
      }
    }
  }
})

export interface CommandListActions {
  onKeyDown: (props: SuggestionKeyDownProps) => boolean
}

// TODO: use predefined command
export interface CommandItem {
  id: string
  name: string
  description?: string
  icon?: React.ReactNode
}

export interface CommandListProps extends SuggestionProps {
  items: CommandItem[]
  onSelectCommand: (command: ChatCommand) => void
}

/**
 * A React component for the command dropdown list.
 * Displays when a user types '/...' and shows available commands.
 */
export const CommandList = forwardRef<CommandListActions, CommandListProps>(
  ({ items: propItems, command, query }, ref) => {
    const [selectedIndex, setSelectedIndex] = useState(0)
    const [isLoading, setIsLoading] = useState(false)
    const [debouncedIsLoading] = useDebounceValue(isLoading, 100)

    // Filter items based on query
    const items = useMemo(() => {
      if (!query) return propItems
      return propItems.filter(item =>
        item.name.toLowerCase().includes(query.toLowerCase())
      )
    }, [propItems, query])

    const handleSelect = (item: CommandItem) => {
      command({
        commandId: item.id,
        label: item.name
      })
    }

    useImperativeHandle(ref, () => ({
      onKeyDown: ({ event }) => {
        if (isLoading) {
          return false
        }
        const lastIndex = items.length - 1
        let newIndex = selectedIndex

        switch (event.key) {
          case 'ArrowUp':
            newIndex = Math.max(0, selectedIndex - 1)
            break
          case 'ArrowDown':
            newIndex = Math.min(lastIndex, selectedIndex + 1)
            break
          case 'Enter':
            if (items[selectedIndex]) {
              handleSelect(items[selectedIndex])
            }
            return true
          default:
            return false
        }

        setSelectedIndex(newIndex)
        return true
      }
    }))

    return (
      <div className="relative flex max-h-[300px] min-w-[60vw] max-w-[90vw] flex-col overflow-hidden rounded-md border bg-background p-1">
        {debouncedIsLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          {items.length === 0 ? (
            <div className="px-2 py-1.5 text-xs text-muted-foreground">
              No commands found
            </div>
          ) : (
            <div className="grid gap-0.5">
              {items.map((item, index) => (
                <CommandItemView
                  key={`${item.id}-${index}`}
                  onClick={() => handleSelect(item)}
                  onMouseEnter={() => setSelectedIndex(index)}
                  title={item.name}
                  isSelected={index === selectedIndex}
                  data={item}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }
)
CommandList.displayName = 'CommandList'

interface CommandItemViewProps extends React.HTMLAttributes<HTMLDivElement> {
  isSelected: boolean
  data: CommandItem
}

function CommandItemView({ isSelected, data, ...rest }: CommandItemViewProps) {
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
        'flex cursor-pointer flex-nowrap items-center gap-1 overflow-hidden rounded-md px-2 py-1.5 text-sm',
        {
          'bg-accent text-accent-foreground': isSelected
        }
      )}
      {...rest}
      ref={ref}
    >
      <span className="mr-2 truncate whitespace-nowrap">/{data.name}</span>
      {data.description && (
        <span className="flex-1 truncate text-xs text-muted-foreground">
          {data.description}
        </span>
      )}
    </div>
  )
}

// TODO: using chat command
export const availableCommands: CommandItem[] = [
  {
    id: 'explain',
    name: 'explain',
    description: 'Explain the selected code'
  },
  {
    id: 'refactor',
    name: 'refactor',
    description: 'Refactor the selected code'
  },
  {
    id: 'test',
    name: 'test',
    description: 'Generate tests for the selected code'
  },
  {
    id: 'docs',
    name: 'docs',
    description: 'Generate documentation for the selected code'
  },
  {
    id: 'fix',
    name: 'fix',
    description: 'Fix issues in the selected code'
  }
]
