'use client'

import React, { useEffect, useRef, useState } from 'react'

import { cn } from '@/lib/utils'

import { Button } from './ui/button'
import { IconChevronDown } from './ui/icons'

interface CollapsibleContainerProps {
  maxHeight?: number
  children: React.ReactNode
}

export const CollapsibleContainer = ({
  maxHeight = 196,
  children
}: CollapsibleContainerProps) => {
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [showButton, setShowButton] = useState(false)
  const contentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (contentRef.current && contentRef.current.scrollHeight > maxHeight) {
      setShowButton(true)
    } else {
      setShowButton(false)
    }
  }, [maxHeight, children])

  const handleToggle = () => {
    setIsCollapsed(!isCollapsed)
  }

  return (
    <div className="relative">
      <div
        ref={contentRef}
        className={cn('h-auto overflow-hidden', {
          'mb-8': showButton && isCollapsed
        })}
        style={{ maxHeight: isCollapsed ? `${maxHeight}px` : 'none' }}
      >
        {children}
      </div>
      {showButton && (
        <div
          className={cn({
            'absolute right-0 -bottom-8 z-10': isCollapsed,
            'flex justify-end my-1': !isCollapsed
          })}
        >
          <Button variant="outline" size="icon" onClick={handleToggle}>
            <IconChevronDown
              className={cn({
                'rotate-180': !isCollapsed
              })}
            />
          </Button>
        </div>
      )}
      {isCollapsed && showButton && (
        <div className="absolute inset-x-0 bottom-0 h-9 bg-gradient-to-t from-background to-transparent"></div>
      )}
    </div>
  )
}
