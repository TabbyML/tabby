'use client'

import React, { useContext, useEffect, useRef, useState } from 'react'

import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { cn } from '@/lib/utils'

import { SectionItem } from '../types'
import { PageContext } from './page-context'

interface Props {
  sections: SectionItem[] | undefined
}

export const Navbar = ({ sections }: Props) => {
  const { isLoading, pendingSectionIds } = useContext(PageContext)
  const [activeNavItem, setActiveNavItem] = useState<string | undefined>()
  const observer = useRef<IntersectionObserver | null>(null)
  const updateActiveNavItem = useDebounceCallback((v: string) => {
    setActiveNavItem(v)
  }, 200)

  useEffect(() => {
    const options = {
      root: null,
      rootMargin: '0px',
      threshold: 0.1
    }

    observer.current = new IntersectionObserver(entries => {
      // Filter and sort entries by boundingClientRect.top
      const sortedEntries = entries
        .filter(entry => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)

      if (sortedEntries.length > 0) {
        const closestEntry = sortedEntries[0]
        updateActiveNavItem.run(closestEntry.target.id)
      }
    }, options)

    const targets = document.querySelectorAll('.section-title')
    targets.forEach(target => {
      observer.current?.observe(target)
    })

    return () => {
      observer.current?.disconnect()
    }
  }, [isLoading, pendingSectionIds])

  return (
    <nav className="sticky right-0 top-0 p-4 pt-8">
      <ul className="flex flex-col space-y-1 border-l">
        {sections?.map(section => {
          const isActive = activeNavItem === section.id
          return (
            <li
              key={section.id}
              className="relative -ml-px cursor-pointer"
              onClick={e => {
                const target = document.getElementById(section.id)
                if (target) {
                  target.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }
              }}
            >
              <div
                className={cn('truncate whitespace-nowrap pl-2 text-sm', {
                  'text-foreground border-l border-foreground': isActive,
                  'text-muted-foreground': !isActive
                })}
              >
                {section.title}
              </div>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
