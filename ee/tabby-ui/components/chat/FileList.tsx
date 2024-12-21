/* eslint-disable no-console */
import React, { useEffect, useRef } from 'react'

import { SuggestionItem } from './types'

interface FileListProps {
  items: SuggestionItem[]
  selectedIndex: number
  onSelect: (item: {
    id: string
    label: string
    category: 'file' | 'symbol'
  }) => void
  onUpdateSelectedIndex: (index: number) => void
}

const MAX_VISIBLE_ITEMS = 4
const ITEM_HEIGHT = 42 // px

export const FileList: React.FC<FileListProps> = ({
  items,
  selectedIndex,
  onSelect,
  onUpdateSelectedIndex
}) => {
  console.log('[FileList] Rendering with items:', items.length)
  console.log('[FileList] Selected index:', selectedIndex)

  const selectedItemRef = useRef<HTMLButtonElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const containerHeight =
    Math.min(items.length, MAX_VISIBLE_ITEMS) * ITEM_HEIGHT

  useEffect(() => {
    const container = containerRef.current
    const selectedItem = selectedItemRef.current
    if (container && selectedItem) {
      const containerTop = container.scrollTop
      const containerBottom = containerTop + container.clientHeight
      const itemTop = selectedItem.offsetTop
      const itemBottom = itemTop + selectedItem.offsetHeight

      if (itemTop < containerTop) {
        container.scrollTop = itemTop
      } else if (itemBottom > containerBottom) {
        container.scrollTop = itemBottom - container.clientHeight
      }
    }
  }, [selectedIndex])

  const renderContent = () => {
    if (!items.length) {
      console.log('[FileList] No items to display')
      return (
        <div className="h-full flex items-center justify-center px-3 py-2.5 text-sm text-muted-foreground/70">
          No files found
        </div>
      )
    }

    return (
      <div
        ref={containerRef}
        className="flex flex-col w-full divide-y divide-border/30 overflow-y-auto"
        style={{
          maxHeight: `${MAX_VISIBLE_ITEMS * ITEM_HEIGHT}px`,
          height: `${containerHeight}px`
        }}
      >
        {items.map((item, index) => {
          console.log(`[FileList] Rendering item: ${item.label}`)
          const filepath =
            'filepath' in item.filepath
              ? item.filepath.filepath
              : item.filepath.uri
          const isSelected = index === selectedIndex

          return (
            <button
              key={filepath}
              ref={isSelected ? selectedItemRef : null}
              onClick={() => {
                console.log(`[FileList] Item selected: ${item.label}`)
                onSelect({
                  id: item.id,
                  label: item.label,
                  category: item.category
                })
              }}
              onMouseEnter={() => {
                console.log(`[FileList] Mouse enter on item: ${item.label}`)
                onUpdateSelectedIndex(index)
              }}
              onMouseDown={e => e.preventDefault()}
              type="button"
              tabIndex={-1}
              style={{ height: `${ITEM_HEIGHT}px` }}
              className={`flex items-center justify-between w-full px-3 text-sm rounded-sm transition-colors flex-shrink-0
                ${
                  isSelected
                    ? 'bg-accent/50 text-accent-foreground'
                    : 'hover:bg-accent/50'
                }
                group relative`}
            >
              <div className="flex items-center gap-2.5 min-w-0 max-w-[60%]">
                <svg
                  className={`w-3.5 h-3.5 shrink-0 ${
                    isSelected
                      ? 'text-accent-foreground'
                      : 'text-muted-foreground/70 group-hover:text-accent-foreground/90'
                  }`}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
                  <polyline points="13 2 13 9 20 9" />
                </svg>
                <span className="font-medium truncate">{item.label}</span>
              </div>
              <span
                className={`text-[11px] truncate max-w-[40%] ${
                  isSelected
                    ? 'text-accent-foreground/90'
                    : 'text-muted-foreground/60 group-hover:text-accent-foreground/80'
                }`}
              >
                {filepath}
              </span>
            </button>
          )
        })}
      </div>
    )
  }

  return <>{renderContent()}</>
}
