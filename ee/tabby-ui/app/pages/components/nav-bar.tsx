'use client'

import React, { useEffect, useRef, useState } from 'react'

import { useDebounceCallback } from '@/lib/hooks/use-debounce'

import { SectionItem } from '../types'

interface Props {
  sections: SectionItem[] | undefined
}

export const Navbar = ({ sections }: Props) => {
  const [activeNavItem, setActiveNavItem] = useState<string | undefined>()
  const observer = useRef<IntersectionObserver | null>(null)
  const updateActiveNavItem = useDebounceCallback((v: string) => {
    setActiveNavItem(v)
  }, 200)

  useEffect(() => {
    const options = {
      root: null,
      rootMargin: '70px'
    }

    observer.current = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          updateActiveNavItem.run(entry.target.id)
          break
        }
      }
    }, options)

    const targets = document.querySelectorAll('.section-title')
    targets.forEach(target => {
      observer.current?.observe(target)
    })

    return () => {
      observer.current?.disconnect()
    }
  }, [])

  return (
    <nav className="sticky right-0 top-0 p-4">
      <ul className="flex flex-col space-y-1">
        {sections?.map(section => (
          <li key={section.id}>
            <div
              className={`truncate whitespace-nowrap text-sm ${
                activeNavItem === section.id
                  ? 'text-foreground'
                  : 'text-muted-foreground'
              }`}
            >
              {section.title}
            </div>
          </li>
        ))}
      </ul>
    </nav>
  )
}
