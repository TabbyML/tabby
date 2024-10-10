'use client'

import { useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import tabbyUrl from '@/assets/logo-dark.png'
import { noop } from 'lodash-es'
import { useQuery } from 'urql'

import { SESSION_STORAGE_KEY } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { useMutation } from '@/lib/tabby/gql'
import { contextInfoQuery } from '@/lib/tabby/query'
import { ThreadRunContexts } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { CardContent, CardFooter } from '@/components/ui/card'
import { IconJetBrains, IconRotate, IconVSCode } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { ClientOnly } from '@/components/client-only'
import { CopyButton } from '@/components/copy-button'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import SlackDialog from '@/components/slack-dialog'
import TextAreaSearch from '@/components/textarea-search'
import { ThemeToggle } from '@/components/theme-toggle'
import { MyAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import { AnimationWrapper } from './components/animation-wrapper'
import Stats from './components/stats'
import { ThreadFeeds } from './components/thread-feeds'

const resetUserAuthTokenDocument = graphql(/* GraphQL */ `
  mutation ResetUserAuthToken {
    resetUserAuthToken
  }
`)

function MainPanel() {
  const { data: healthInfo } = useHealth()
  const [{ data }] = useMe()
  const isChatEnabled = useIsChatEnabled()
  const [isShowDemoBanner] = useShowDemoBanner()
  const elementRef = useRef<HTMLDivElement | null>(null)
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [{ data: contextInfoData, fetching: fetchingContextInfo }] = useQuery({
    query: contextInfoQuery
  })

  // Prefetch the search page
  useEffect(() => {
    router.prefetch('/search')
  }, [router])

  // FIXME add loading skeleton
  if (!healthInfo || !data?.me) return <></>

  const onSearch = (question: string, ctx?: ThreadRunContexts) => {
    setIsLoading(true)
    sessionStorage.setItem(SESSION_STORAGE_KEY.SEARCH_INITIAL_MSG, question)
    sessionStorage.setItem(
      SESSION_STORAGE_KEY.SEARCH_INITIAL_CONTEXTS,
      JSON.stringify(ctx)
    )
    router.push('/search')
  }

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }

  return (
    <ScrollArea style={style}>
      <header className="flex h-16 items-center justify-end px-4 backdrop-blur lg:px-10 sticky top-0 z-10">
        <div className="flex items-center gap-x-6">
          <ClientOnly>
            <ThemeToggle />
          </ClientOnly>
          <UserPanel showHome={false} showSetting>
            <MyAvatar className="h-10 w-10 border" />
          </UserPanel>
        </div>
      </header>

      <main
        className="flex-col items-center justify-center pb-8 lg:flex lg:pb-0"
        ref={elementRef}
      >
        <div className="mx-auto flex min-h-0 w-full flex-col items-center px-10 lg:-mt-[2vh] lg:max-w-4xl lg:px-0">
          <AnimationWrapper
            viewport={{
              margin: '-120px 0px 0px 0px'
            }}
          >
            <Image
              src={tabbyUrl}
              alt="logo"
              width={192}
              className={cn('mt-4 invert dark:invert-0', {
                'mb-4': isChatEnabled,
                'mb-2': !isChatEnabled
              })}
            />
          </AnimationWrapper>
          {isChatEnabled && (
            <AnimationWrapper
              viewport={{ margin: '-180px 0px 0px 0px' }}
              style={{ width: '100%' }}
            >
              <div className="mb-6">
                <TextAreaSearch
                  onSearch={onSearch}
                  showBetaBadge
                  autoFocus
                  loadingWithSpinning
                  isLoading={isLoading}
                  cleanAfterSearch={false}
                  contextInfo={contextInfoData?.contextInfo}
                  fetchingContextInfo={fetchingContextInfo}
                  // className="min-h-[7rem]"
                />
              </div>
            </AnimationWrapper>
          )}
          <Stats />
          <ThreadFeeds className="w-full" />
        </div>
      </main>
    </ScrollArea>
  )
}

// todo move to header
function Configuration({ className }: { className?: string }) {
  const [{ data }, reexecuteQuery] = useMe()
  const externalUrl = useExternalURL()

  const resetUserAuthToken = useMutation(resetUserAuthTokenDocument, {
    onCompleted: () => reexecuteQuery()
  })

  if (!data?.me) return <></>

  return (
    <div className={className}>
      <CardContent className="flex flex-col gap-6 px-0">
        <div className="flex flex-col">
          <span className="flex items-center gap-1">
            <Label className="text-xs font-semibold">Endpoint URL</Label>
            <CopyButton value={externalUrl} />
          </span>
          <span className="flex items-center gap-1">
            <Input
              value={externalUrl}
              onChange={noop}
              className="h-7 max-w-[320px] rounded-none border-x-0 !border-t-0 border-muted-foreground p-0 shadow-none dark:border-primary/50"
            />
          </span>
        </div>

        <div className="flex flex-col">
          <span className="flex items-center gap-1">
            <Label className="text-xs font-semibold">Token</Label>
            <CopyButton value={data.me.authToken} />
            <Button
              title="Rotate"
              size="icon"
              variant="hover-destructive"
              onClick={() => resetUserAuthToken()}
            >
              <IconRotate />
            </Button>
          </span>
          <span className="flex items-center gap-1">
            <Input
              className="h-7 max-w-[320px] rounded-none border-x-0 !border-t-0 border-muted-foreground p-0 font-mono shadow-none dark:border-primary/50"
              value={data.me.authToken}
              onChange={noop}
            />
          </span>
        </div>
      </CardContent>
      <CardFooter className="px-0 pb-2 text-xs text-muted-foreground">
        <span>
          Use information above for IDE extensions / plugins configuration, see{' '}
          <a
            className="underline"
            target="_blank"
            href="https://tabby.tabbyml.com/docs/extensions/configurations#server"
          >
            documentation website
          </a>{' '}
          for details
        </span>
      </CardFooter>

      <div className="mb-6 mt-3 flex gap-x-3 lg:mb-0">
        <IDELink
          href="https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby"
          name="Visual Studio Code"
          icon={<IconVSCode className="h-5 w-5" />}
        />
        <IDELink
          href="https://plugins.jetbrains.com/plugin/22379-tabby"
          name="JetBrains"
          icon={<IconJetBrains className="h-5 w-5" />}
        />
      </div>
    </div>
  )
}

function IDELink({
  href,
  name,
  icon
}: {
  href: string
  name: string
  icon: React.ReactNode
}) {
  return (
    <Tooltip>
      <TooltipTrigger>
        <Link
          href={href}
          className="transition-all hover:opacity-80 dark:text-muted-foreground"
          target="_blank"
        >
          {icon}
        </Link>
      </TooltipTrigger>
      <TooltipContent>
        <p>{name}</p>
      </TooltipContent>
    </Tooltip>
  )
}

export default function Home() {
  return (
    <div>
      <MainPanel />
      <SlackDialog />
    </div>
  )
}
