'use client'

import * as React from 'react'

import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconGitHub, IconExternalLink } from '@/components/ui/icons'
import dynamic from 'next/dynamic'
import Link from 'next/link'

const ThemeToggle = dynamic(
  () => import('@/components/theme-toggle').then(x => x.ThemeToggle),
  { ssr: false }
)

export function Header() {
  const [isChatEnabled, setIsChatEnabled] = React.useState(false);
  React.useEffect(() => {
    fetchIsChatEnabled().then(setIsChatEnabled);
  }, []);
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

async function fetchIsChatEnabled() {
  const resp = await fetch("/v1/health");
  const json = await resp.json();
  return !!json.chat_model;
}