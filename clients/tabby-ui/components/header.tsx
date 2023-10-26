'use client'

import * as React from 'react'

import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconGitHub, IconExternalLink, IconNotice } from '@/components/ui/icons'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useHealth } from '@/lib/hooks/use-health'
import { useLatestRelease } from '@/lib/hooks/use-latest-release'
import { compare } from 'compare-versions'

const ThemeToggle = dynamic(
  () => import('@/components/theme-toggle').then(x => x.ThemeToggle),
  { ssr: false }
)

export function Header() {
  const { data } = useHealth();
  const isChatEnabled = !!data?.chat_model;
  const version = data?.version?.git_describe;
  const { data: latestRelease } = useLatestRelease();
  const newVersionAvailable = version && latestRelease && compare(latestRelease.name, version, '>');

  return (
    <header className="sticky top-0 z-50 flex items-center justify-between w-full h-16 px-4 border-b shrink-0 bg-gradient-to-b from-background/10 via-background/50 to-background/80 backdrop-blur-xl">
      <div className="flex items-center">
        <ThemeToggle />
        <Link href="/" className={cn(buttonVariants({ variant: 'link' }))}>
          Home
        </Link>
        {isChatEnabled && <Link href="/playground" className={cn(buttonVariants({ variant: 'link' }))}>
          Playground
        </Link>}
      </div>
      <div className="flex items-center justify-end space-x-2">
        {newVersionAvailable && <a
          target="_blank"
          href="https://github.com/TabbyML/tabby/releases/latest"
          rel="noopener noreferrer"
          className={buttonVariants({ variant: 'ghost' })}
        >
          <IconNotice className='text-yellow-600 dark:text-yellow-400' />
          <span className="hidden ml-2 md:flex">New version ({latestRelease.name}) available</span>
        </a>}
        <a
          target="_blank"
          href="https://github.com/TabbyML/tabby"
          rel="noopener noreferrer"
          className={cn(buttonVariants({ variant: 'outline' }))}
        >
          <IconGitHub />
          <span className="hidden ml-2 md:flex">GitHub</span>
        </a>
        <a
          target="_blank"
          href="/swagger-ui"
          rel="noopener noreferrer"
          className={cn(buttonVariants({ variant: 'outline' }))}
        >
          <IconExternalLink />
          <span className="hidden ml-2 md:flex">OpenAPI</span>
        </a>
      </div>
    </header>
  )
}