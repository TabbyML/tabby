import Mention from '@tiptap/extension-mention'
import { NodeViewWrapper, ReactNodeViewRenderer } from '@tiptap/react'
import { Filepath } from 'tabby-chat-panel/index'

import { cn } from '@/lib/utils'

import type { FileItem, MentionNodeAttrs, SourceItem } from './types'

// A regular expression to match fileItem placeholders in the text content.
// For example: [[fileItem:{"label":"some/file/path","id":"123"}]]
export const FILEITEM_REGEX = /\[\[fileItem:({.*?})\]\]/g

/**
 * Converts a FileItem to a SourceItem for the mention list.
 */
export function fileItemToSourceItem(info: FileItem): SourceItem {
  return {
    fileItem: info,
    // The name is the filename (e.g., split by '/')
    name: info.label.split('/').pop() || info.label,
    // The filepath is the original label (which might be a path)
    filepath: info.label,
    category: 'file'
  }
}

/**
 * A small file icon (SVG) for indicating file items.
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
 * Shortens a file name or string, preserving only the last suffixLength characters.
 */
export function shortenLabel(label: string, suffixLength = 15): string {
  if (label.length <= suffixLength) return label
  return '...' + label.slice(label.length - suffixLength)
}

/**
 * A React component that renders a mention node inside the editor.
 * It shows a small file icon and file name in a highlighted style.
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
 * When rendered as text, it produces a placeholder with file info.
 */
export const PromptFormMentionExtension = Mention.extend({
  // Use a React-based node view for the mention
  addNodeView() {
    return ReactNodeViewRenderer(MentionComponent)
  },

  // How the mention node is rendered as plain text (e.g. exporting final content)
  renderText({ node }) {
    const fileItem = node.attrs.fileItem as FileItem
    return `[[fileItem:${JSON.stringify(fileItem)}]]`
  },

  // Define attributes stored in the mention node
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
 * A React component rendering the mention dropdown list.
 * Shown when the user types '@...' and possible file completions are fetched.
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
   * Handle user selecting an item from the mention list.
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
        // Prevent the popover from closing on clicks
        e.preventDefault()
      }}
    >
      {/* If no items found, show a simple message */}
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

// Some utils function help to extract place holder
export function replaceAtMentionPlaceHolderWithAt(value: string) {
  // eslint-disable-next-line no-console
  console.log('before value: ', value)

  let newValue = value

  let match
  while ((match = FILEITEM_REGEX.exec(value)) !== null) {
    try {
      const parsedItem = JSON.parse(match[1])
      const labelName =
        parsedItem.label.split('/').pop() || parsedItem.label || 'unknown'
      newValue = newValue.replace(match[0], `@${labelName}`)
    } catch (error) {
      continue
    }
  }

  // eslint-disable-next-line no-console
  console.log('new value:', newValue)
  return newValue
}

interface ReplaceResult {
  newValue: string
  fileItems: FileItem[]
}

export const FILEITEM_AT_REGEX = /\[\[fileItemAt: (\d+)\]\]/g

// Some utils function help to extract place holder
// replace at mention JSON placeholder to something like [[fileItemAt: id]] which ad is string
// return a list of FileItem with unique id
// also return string already replaced
// example:
//  [[fileItem:{"label":"src/CodeActions.ts","filepath":{"kind":"git","filepath":"clients/vscode/src/CodeActions.ts",
// "gitUrl":"git@github.com:Sma1lboy/tabby.git"}}]] explain this  [[fileItem:{"label":"src/CodeActions.ts",
// "filepath":{"kind":"git","filepath":"clients/vscode/src/CodeActions.ts","gitUrl":"git@github.com:Sma1lboy/tabby.git"}}]]
// will replaced as [[fileItemAt: idx0]] explain this [[fileItemAt: idx1]]
// and with [fileItem1, fileItem2]
export function replaceAtMentionPlaceHolderWithAtPlaceHolder(
  value: string
): ReplaceResult {
  // eslint-disable-next-line no-console
  console.log('before value: ', value)

  let newValue = value
  let match
  const fileItems: FileItem[] = []
  let idx = 0

  while ((match = FILEITEM_REGEX.exec(value)) !== null) {
    try {
      const parsedItem = JSON.parse(match[1])

      fileItems.push({
        ...parsedItem
      })

      newValue = newValue.replace(match[0], `[[fileItemAt: ${idx}]]`)

      idx++
    } catch (error) {
      continue
    }
  }

  // eslint-disable-next-line no-console
  console.log('new value:', newValue)
  return {
    newValue,
    fileItems
  }
}

// utils function
export function getFilepathStringByChatPanelFilePath(
  filepath: Filepath
): string {
  return 'filepath' in filepath ? filepath.filepath : filepath.uri
}

export function getLastSegmentFromPath(filepath: string): string {
  if (!filepath) {
    return 'unknown'
  }

  const normalizedPath = filepath.replace(/\\/g, '/')

  const cleanPath = normalizedPath.replace(/\/+$/, '')
  const segments = cleanPath.split('/')
  const lastSegment = segments[segments.length - 1]

  return lastSegment || 'unknown'
}
