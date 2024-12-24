import React, { useEffect, useRef } from 'react'

import { MentionNodeAttrs, SourceItem } from './prompt-form-editor/types'

interface PopoverMentionListProps {
  items: SourceItem[]
  selectedIndex: number
  onUpdateSelectedIndex: (index: number) => void
  handleItemSelection: (
    item: SourceItem,
    command?: (props: MentionNodeAttrs) => void
  ) => void
}

// Maximum number of items visible in the list
const MAX_VISIBLE_ITEMS = 4
// Height of each item in pixels
const ITEM_HEIGHT = 42

export const PopoverMentionList: React.FC<PopoverMentionListProps> = ({
  items,
  selectedIndex,
  onUpdateSelectedIndex,
  handleItemSelection
}) => {
  const selectedItemRef = useRef<HTMLButtonElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Dynamically calculate container height based on number of items
  const containerHeight =
    Math.min(items.length, MAX_VISIBLE_ITEMS) * ITEM_HEIGHT

  // Scroll into view for the currently selected item
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

  // Render list content
  const renderContent = () => {
    if (!items.length) {
      return (
        <div className="text-muted-foreground/70 flex h-full items-center justify-center px-3 py-2.5 text-sm">
          No files found
        </div>
      )
    }

    return (
      <div
        ref={containerRef}
        className="divide-border/30 flex w-full flex-col divide-y overflow-y-auto"
        style={{
          maxHeight: `${MAX_VISIBLE_ITEMS * ITEM_HEIGHT}px`,
          height: `${containerHeight}px`
        }}
      >
        {items.map((item, index) => {
          const filepath = item.filepath
          const isSelected = index === selectedIndex

          return (
            <button
              key={filepath + '-' + item.name}
              ref={isSelected ? selectedItemRef : null}
              onClick={() => {
                handleItemSelection(item)
              }}
              onMouseEnter={() => {
                onUpdateSelectedIndex(index)
              }}
              onMouseDown={e => e.preventDefault()}
              type="button"
              tabIndex={-1}
              style={{ height: `${ITEM_HEIGHT}px` }}
              className={`flex w-full shrink-0 items-center justify-between rounded-sm px-3 text-sm transition-colors
                ${
                  isSelected
                    ? 'bg-accent/50 text-accent-foreground'
                    : 'hover:bg-accent/50'
                }
                group relative`}
            >
              <div className="flex min-w-0 max-w-[60%] items-center gap-2.5">
                <svg
                  className={`h-3.5 w-3.5 shrink-0 ${
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
                <span className="truncate font-medium">{item.name}</span>
              </div>
              <span
                className={`max-w-[40%] truncate text-[11px] ${
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
