import React, { useEffect, useMemo, useRef, useState } from 'react'
import { compact } from 'lodash-es'

import { useDebounceCallback } from '@/lib/hooks/use-debounce'

import { ConversationPair } from './page'

interface Props {
  qaPairs: ConversationPair[] | undefined
}

export const Navbar = ({ qaPairs }: Props) => {
  const sections = useMemo(() => {
    if (!qaPairs?.length) return []
    return compact(qaPairs.map(x => x.question))
  }, [qaPairs])

  const [activeNavItem, setActiveNavItem] = useState<string | undefined>()
  const observer = useRef<IntersectionObserver | null>(null)
  const updateActiveNavItem = useDebounceCallback((v: string) => {
    setActiveNavItem(v)
  }, 200)

  useEffect(() => {
    const options = {
      root: null,
      rootMargin: '70px'
      // threshold: 0.5,
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
        {sections.map(section => (
          <li key={section.id}>
            <div
              className={`truncate whitespace-nowrap text-sm ${
                activeNavItem === section.id
                  ? 'text-foreground'
                  : 'text-muted-foreground'
              }`}
            >
              {section.content}
            </div>
          </li>
        ))}
      </ul>
    </nav>
  )
}
