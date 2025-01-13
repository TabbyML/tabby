import Mention from '@tiptap/extension-mention'
import { NodeViewWrapper, ReactNodeViewRenderer } from '@tiptap/react'

import { cn } from '@/lib/utils'

import type { FileItem, MentionNodeAttrs, SourceItem } from './types'
import { shortenLabel } from './utils'

/**
 * A small file icon component to indicate file items in a mention.
 */
export const FileItemIcon = () => (
  <svg
    className="h-4 w-4 text-muted-foreground"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
  >
    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
    <polyline points="13 2 13 9 20 9" />
  </svg>
)

/**
 * A React component to render a mention node in the editor.
 * Displays the filename and an icon in a highlighted style.
 */
export const MentionComponent = ({ node }: { node: any }) => {
  return (
    <NodeViewWrapper className="inline-block align-middle -my-1">
      <span
        className={cn(
          'bg-muted prose inline-flex items-center rounded px-1.5 py-0.5 text-sm font-medium text-white',
          'ring-muted ring-1 ring-inset',
          'relative top-[0.1em]'
        )}
        data-category={node.attrs.category}
      >
        <FileItemIcon />
        <span className="relative top-[-0.5px]">{node.attrs.name}</span>
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
    const fileItem = node.attrs.fileItem as FileItem
    return `[[fileItem:${JSON.stringify(fileItem)}]]`
  },

  // Defines custom attributes for the mention node
  addAttributes() {
    return {
      id: {
        default: null,
        parseHTML: element => element.getAttribute('data-id'),
        renderHTML: attrs => {
          if (!attrs.id) return {}
          return { 'data-id': attrs.id }
        }
      },
      name: {
        default: null,
        parseHTML: element => element.getAttribute('data-name'),
        renderHTML: attrs => {
          if (!attrs.name) return {}
          return { 'data-name': attrs.name }
        }
      },
      category: {
        default: 'file',
        parseHTML: element => element.getAttribute('data-category'),
        renderHTML: attrs => {
          if (!attrs.category) return {}
          return { 'data-category': attrs.category }
        }
      },
      fileItem: {
        default: null,
        parseHTML: element => element.getAttribute('data-fileItem'),
        renderHTML: attrs => {
          if (!attrs.fileItem) return {}
          return { 'data-fileItem': attrs.fileItem }
        }
      }
    }
  }
})

/**
 * A React component for the mention dropdown list.
 * Displays when a user types '@...' and suggestions are fetched.
 */
export const MentionList = ({
  items,
  command,
  selectedIndex,
  onHover
}: {
  items: SourceItem[]
  command: ((props: MentionNodeAttrs) => void) | null
  selectedIndex: number
  onHover: (index: number) => void
}) => {
  /**
   * Handle the user selecting an item from the mention list.
   */
  const handleSelect = (item: SourceItem) => {
    if (!command) return
    command({
      id: `${item.name}-${item.filepath}`,
      name: item.name,
      category: 'file',
      fileItem: item.fileItem
    })
  }

  return (
    <div
      className="max-h-[300px] overflow-auto p-1"
      onMouseDown={e => {
        // Prevent the dropdown from closing when clicked
        e.preventDefault()
      }}
    >
      {/* If no items are found, show a message. */}
      {!items.length ? (
        <div className="px-2 py-1.5 text-sm text-muted-foreground">
          Cannot find any files.
        </div>
      ) : (
        <div className="grid gap-0.5">
          {items.map((item, index) => (
            <button
              key={index}
              className={cn(
                'flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm',
                index === selectedIndex
                  ? 'bg-accent text-accent-foreground'
                  : 'hover:bg-accent hover:text-accent-foreground'
              )}
              onMouseEnter={() => onHover(index)}
              onMouseDown={e => {
                e.preventDefault()
                handleSelect(item)
              }}
            >
              <FileItemIcon />
              <span className="flex-1 truncate">{shortenLabel(item.name)}</span>
              <span className="max-w-[150px] truncate text-xs text-muted-foreground">
                {shortenLabel(item.filepath, 20)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
