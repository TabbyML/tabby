'use client'

import React from 'react'
import { isNil } from 'lodash-es'
import useSWRImmutable from 'swr/immutable'

import { useIsDemoMode } from '@/lib/hooks/use-server-info'
import { cn } from '@/lib/utils'
import {
  IconClose,
  IconGitFork,
  IconGithub,
  IconStar
} from '@/components/ui/icons'

export const BANNER_HEIGHT = '3.5rem'

interface ShowDemoBannerContextValue {
  isShowDemoBanner: boolean
  setIsShowDemoBanner: React.Dispatch<React.SetStateAction<boolean>>
}

const ShowDemoBannerContext = React.createContext<ShowDemoBannerContextValue>(
  {} as ShowDemoBannerContextValue
)

export const ShowDemoBannerProvider = ({
  children
}: {
  children: React.ReactNode
}) => {
  const isDemoMode = useIsDemoMode()
  const [isShowDemoBanner, setIsShowDemoBanner] = React.useState(false)

  React.useEffect(() => {
    const isInIframe = window.self !== window.top
    if (isInIframe) return

    if (!isNil(isDemoMode)) {
      setIsShowDemoBanner(isDemoMode)
    }
  }, [isDemoMode])

  return (
    <ShowDemoBannerContext.Provider
      value={{ isShowDemoBanner, setIsShowDemoBanner }}
    >
      {children}
    </ShowDemoBannerContext.Provider>
  )
}

export function useShowDemoBanner(): [
  boolean,
  React.Dispatch<React.SetStateAction<boolean>>
] {
  const { isShowDemoBanner, setIsShowDemoBanner } = React.useContext(
    ShowDemoBannerContext
  )
  return [isShowDemoBanner, setIsShowDemoBanner]
}

export function DemoBanner() {
  const [isShowDemoBanner, setIsShowDemoBanner] = useShowDemoBanner()
  const { data } = useSWRImmutable(
    'https://api.github.com/repos/TabbyML/tabby',
    (url: string) => fetch(url).then(res => res.json())
  )
  const style = isShowDemoBanner ? { height: BANNER_HEIGHT } : { height: 0 }
  return (
    <div
      className={cn(
        'flex items-center justify-between bg-primary px-4 text-primary-foreground transition-all md:px-5',
        {
          'opacity-100 pointer-events-auto': isShowDemoBanner,
          'opacity-0 pointer-events-none': !isShowDemoBanner
        }
      )}
      style={style}
    >
      <a
        href="https://links.tabbyml.com/schedule-a-demo"
        target="_blank"
        className="flex items-center gap-x-2 text-xs font-semibold underline md:text-sm"
      >
        <span>📆</span>
        <span>Book a 30-minute product demo.</span>
      </a>

      {isShowDemoBanner && (
        <img
          referrerPolicy="no-referrer-when-downgrade"
          src="https://static.scarf.sh/a.png?x-pxid=b1d0308a-b3c5-425a-972a-378d883a2bca"
        />
      )}

      <div className="flex items-center gap-x-4 md:gap-x-8">
        <a
          href="https://github.com/TabbyML/tabby"
          target="_blank"
          className="flex items-center transition-all hover:opacity-70"
        >
          <IconGithub />
          <div className="ml-2 hidden md:block">
            <p className="text-xs font-semibold">TabbyML/tabby</p>
            <div
              className={cn('flex items-center text-xs transition-all', {
                'h-4 opacity-70': data,
                'h-0 opacity-0': !data
              })}
            >
              <IconStar className="mr-1 h-2.5 w-2.5" />
              <span>{data?.stargazers_count}</span>
              <IconGitFork className="ml-2 mr-1 h-2.5 w-2.5" />
              <span>{data?.forks_count}</span>
            </div>
          </div>
        </a>

        <IconClose
          className="cursor-pointer transition-all hover:opacity-70"
          onClick={() => setIsShowDemoBanner(false)}
        />
      </div>
    </div>
  )
}
