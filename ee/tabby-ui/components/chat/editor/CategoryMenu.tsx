/* eslint-disable no-console */
import React from 'react'

interface CategoryMenuProps {
  items: {
    label: string
    category: 'file' | 'symbol'
  }[]
  selectedIndex: number
  onSelect: (category: 'file' | 'symbol') => void
  onUpdateSelectedIndex: (index: number) => void
}

export const CategoryMenu: React.FC<CategoryMenuProps> = ({
  items,
  selectedIndex,
  onSelect,
  onUpdateSelectedIndex
}) => {
  console.log('[CategoryMenu] Rendering component with items:', items)

  return (
    <div className="flex flex-col w-full h-full divide-y divide-border/30">
      {items.map((item, idx) => {
        const isSelected = idx === selectedIndex

        const handleMouseEnter = () => onUpdateSelectedIndex(idx)

        const handleClick = () => {
          console.log(`[CategoryMenu] ${item.category} category selected`)
          onSelect(item.category)
        }

        return (
          <button
            key={item.label}
            onClick={handleClick}
            onMouseEnter={handleMouseEnter}
            className={`flex items-center px-3 py-2.5 text-sm transition-colors 
              ${isSelected ? 'bg-accent/50' : 'hover:bg-accent/50'}
            `}
          >
            <div className="flex items-center gap-2.5">
              <svg
                className="w-3.5 h-3.5 text-muted-foreground/70"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                {item.category === 'file' ? (
                  <>
                    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
                    <polyline points="13 2 13 9 20 9" />
                  </>
                ) : (
                  <>
                    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                  </>
                )}
              </svg>
              <span className="font-medium">{item.label}</span>
            </div>
          </button>
        )
      })}
    </div>
  )
}
