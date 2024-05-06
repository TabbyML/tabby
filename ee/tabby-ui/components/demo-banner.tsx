'use client'

import { useState, useEffect } from 'react'
import { isNil } from 'lodash-es'
import useSWRImmutable from 'swr/immutable'
import { useTheme } from 'next-themes'

import { cn } from '@/lib/utils'
import { useIsDemoMode } from '@/lib/hooks/use-server-info'
import { IconClose, IconSlack, IconStar, IconGitFork, IconGithub } from '@/components/ui/icons'

export const BANNER_HEIGHT = '3.5rem'

export function useShowDemoBanner (): [boolean, (isShow: boolean) => void] {
  const isDemoMode = useIsDemoMode()
  const [isShow, setIsShow] = useState(false)

  useEffect(() => {
    if (!isNil(isDemoMode)) {
      setIsShow(isDemoMode)
    }
  }, [isDemoMode])

  return [isShow, setIsShow]
}

export function DemoBanner () {
  const [isShow, setIsShow] = useShowDemoBanner()
  const [slackIconFill, setSlackIconFill] = useState('')
  const { theme } = useTheme()
  const { data } = useSWRImmutable(
    "https://api.github.com/repos/TabbyML/tabby",
    (url: string) => fetch(url).then(res => res.json())
  )
  
  useEffect(() => {
    setSlackIconFill(theme === 'dark' ? '#171615' : '#ECECEC')
  }, [isShow, theme])

  const style = isShow
    ? { height: BANNER_HEIGHT }
    : { height: 0 }
  return (
    <div
      className={cn("flex items-center justify-between bg-primary px-4 text-primary-foreground transition-all md:px-5", {
        "opacity-100 pointer-events-auto": isShow,
        "opacity-0 pointer-events-none": !isShow
      })}
      style={style}>
      <a
        href="https://links.tabbyml.com/schedule-a-demo"
        target="_blank"
        className="flex items-center gap-x-2 text-xs font-semibold hover:underline md:text-sm"
      >
        <span>📆</span>
        <span>Book a 30-minute product demo</span>
      </a>

      <div className="flex items-center gap-x-4 md:gap-x-8">
        <a
          href="https://links.tabbyml.com/join-slack"
          target="_blank"
          className="flex items-center transition-all hover:opacity-70">
          <IconSlack className="h-7 w-7" fill={slackIconFill} />
          <span className="hidden text-xs font-semibold md:block">Join Slack</span>
        </a>
        
        <a
          href="https://github.com/TabbyML/tabby"
          target="_blank"
          className="flex items-center transition-all hover:opacity-70">
          <IconGithub />
          <div className="ml-2 hidden md:block">
            <p className="text-xs font-semibold">TabbyML/tabby</p>
            <div className={cn("flex items-center text-xs transition-all", {
              'h-4 opacity-70': data,
              'h-0 opacity-0': !data
            })}>
              <IconStar className="mr-1 h-2.5 w-2.5" />
              <span>{data?.stargazers_count}</span>
              <IconGitFork className="ml-2 mr-1 h-2.5 w-2.5" />
              <span>{data?.forks_count}</span>
            </div>
          </div>
        </a>
        
        <IconClose className="cursor-pointer transition-all hover:opacity-70" onClick={() => setIsShow(false)} />
      </div>
    </div>
  )
}