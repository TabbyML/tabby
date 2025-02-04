import React, {
  forwardRef,
  HTMLAttributes,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import Mention from '@tiptap/extension-mention'
import { NodeViewWrapper, ReactNodeViewRenderer } from '@tiptap/react'
import { SuggestionKeyDownProps, SuggestionProps } from '@tiptap/suggestion'
import { uniqBy } from 'lodash-es'
import { FileText, SquareFunctionIcon } from 'lucide-react'
import {
  Filepath,
  ListActiveSymbolItem,
  ListFileItem,
  ListFilesInWorkspaceParams
} from 'tabby-chat-panel/index'

import { cn, convertFilepath, resolveFileNameForDisplay } from '@/lib/utils'
import { IconChevronLeft } from '@/components/ui/icons'

import { emitter } from '../event-emitter'
import type { CategoryItem, CategoryMenu, FileItem, SourceItem } from './types'
import {
  fileItemToSourceItem,
  filterItemsByQuery,
  symbolItemToSourceItem
} from './utils'

/**
 * A React component to render a mention node in the editor.
 * Displays the filename and an icon in a highlighted style.
 */
export const MentionComponent = ({ node }: { node: any }) => {
  const { category, fileItem, label } = node.attrs

  // FIXME(@jueliang) fine a better way to detect the mention
  useEffect(() => {
    emitter.emit('file_mention_update')

    return () => {
      emitter.emit('file_mention_update')
    }
  }, [])
  // eslint-disable-next-line no-console
  console.log('node attrs', node.attrs)
  return (
    <NodeViewWrapper as="span" className="rounded-sm px-1">
      <span
        className={cn(
          'space-x-0.5 whitespace-nowrap rounded bg-muted px-1.5 py-0.5 align-middle text-sm font-medium text-foreground'
        )}
        data-category={category}
      >
        <>
          {category === 'file' ? (
            <FileText className="relative -top-px inline-block h-3.5 w-3.5" />
          ) : category === 'symbol' ? (
            <SquareFunctionIcon className="relative -top-px inline-block h-3.5 w-3.5" />
          ) : null}
          <span className="relative whitespace-normal">{label}</span>
        </>
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
    const category = node.attrs.category

    // If symbols can be mentioned later, the placeholder could be [[symbol:{label}]].
    switch (category) {
      case 'symbol':
        return `[[symbol:${JSON.stringify(node.attrs.fileItem)}]]`
      case 'file':
      default:
        return `[[file:${JSON.stringify(filePath)}]]`
    }
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
      },
      // label could be basename of path or symbol name
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

export interface MentionListActions {
  onKeyDown: (props: SuggestionKeyDownProps) => boolean
}

export interface MentionListProps extends SuggestionProps {
  items: SourceItem[]
  listFileInWorkspace?: (
    params: ListFilesInWorkspaceParams
  ) => Promise<ListFileItem[]>
  listActiveSymbols?: () => Promise<ListActiveSymbolItem[]>
  onSelectItem: (item: SourceItem) => void
}

/**
 * A React component for the mention dropdown list.
 * Displays when a user types '@...' and suggestions are fetched.
 */
export const MentionList = forwardRef<MentionListActions, MentionListProps>(
  (
    {
      items: propItems,
      command,
      query,
      listFileInWorkspace,
      listActiveSymbols
    },
    ref
  ) => {
    const [items, setItems] = useState<SourceItem[]>(propItems)
    const [selectedIndex, setSelectedIndex] = useState(0)
    const [mode, setMode] = useState<CategoryMenu>('category')
    const categories = useMemo(
      () =>
        [
          {
            label: 'Files',
            category: 'file',
            icon: <FileText className="w-4 h-4" />
          },
          {
            label: 'Symbols',
            category: 'symbol',
            icon: <SquareFunctionIcon className="w-4 h-4" />
          }
        ] as CategoryItem[],
      []
    )

    const handleSelect = (item: SourceItem) => {
      if (item.isRootCategoryItem) {
        setMode(item.category)
        return
      }

      let label = item.name
      if (item.category === 'file') {
        label = resolveFileNameForDisplay(
          convertFilepath(item.fileItem.filepath).filepath || ''
        )
      }

      command({
        category: item.category,
        fileItem: item.fileItem,
        label: label
      })
    }

    useEffect(() => setSelectedIndex(0), [items])

    useEffect(() => {
      const fetchOptions = async () => {
        if (mode === 'category') {
          if (query) {
            const files = (await listFileInWorkspace?.({ query })) || []
            setItems(files.map(fileItemToSourceItem))
            return
          }

          const [files] = await Promise.all([
            listFileInWorkspace?.({ query: '' }) || []
          ])
          setItems([
            ...categories.map(
              c =>
                ({
                  id: c.type,
                  name: c.label,
                  filepath: '',
                  category: c.category,
                  isRootCategoryItem: true,
                  fileItem: {} as FileItem,
                  icon: c.icon
                } as SourceItem)
            ),
            ...files.map(fileItemToSourceItem)
          ])
          return
        }

        if (mode === 'file') {
          const files = (await listFileInWorkspace?.({ query })) || []
          setItems(files.map(fileItemToSourceItem))
        } else {
          const symbols = await listActiveSymbols?.()
          const symbolItems = uniqBy(symbols?.map(symbolItemToSourceItem), 'id')
          setItems(filterItemsByQuery(symbolItems, query))
        }
      }

      fetchOptions()
    }, [categories, listActiveSymbols, listFileInWorkspace, mode, query])

    useImperativeHandle(ref, () => ({
      onKeyDown: ({ event }) => {
        const lastIndex = items.length - 1
        let newIndex = selectedIndex

        switch (event.key) {
          case 'ArrowUp':
            newIndex = selectedIndex > 0 ? selectedIndex - 1 : lastIndex
            break
          case 'ArrowDown':
            newIndex = selectedIndex < lastIndex ? selectedIndex + 1 : 0
            break
          case 'Enter':
            if (items[selectedIndex]) {
              handleSelect(items[selectedIndex])
              if (items[selectedIndex].isRootCategoryItem) {
                setSelectedIndex(0)
              }
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
      <div className="flex max-h-[300px] min-w-[60vw] max-w-[90vw] flex-col overflow-hidden rounded-md border bg-background p-1">
        {mode !== 'category' && (
          <div className="text-muted-foreground flex items-center  p-1 text-sm">
            <button
              className="hover:bg-accent mr-2 rounded p-1"
              onClick={() => setMode('category')}
            >
              <IconChevronLeft className="h-4 w-4" />
            </button>
            {mode === 'file' ? 'Files' : 'Symbols'}
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          {items.length === 0 ? (
            <div className="text-muted-foreground px-2 py-1.5 text-xs">
              {/* If no items are found, show a message. */}
              {query ? 'No results found' : 'Type to search...'}
            </div>
          ) : (
            <div className="grid gap-0.5">
              {items.map((item, index) => (
                <OptionItemView
                  key={item.id + '-' + index}
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
MentionList.displayName = 'MentionList'

interface OptionItemView extends HTMLAttributes<HTMLDivElement> {
  isSelected: boolean
  data: SourceItem
}
function OptionItemView({ isSelected, data, ...rest }: OptionItemView) {
  const ref = useRef<HTMLDivElement>(null)
  const filepathWithoutFilename = useMemo(() => {
    return data.filepath.split('/').slice(0, -1).join('/')
  }, [data.filepath])

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
      <span className="flex h-5 shrink-0 items-center">{data.icon}</span>
      <span className="mr-2 truncate whitespace-nowrap">{data.name}</span>
      <span className="flex-1 truncate text-xs text-muted-foreground">
        {filepathWithoutFilename}
      </span>
    </div>
  )
}
