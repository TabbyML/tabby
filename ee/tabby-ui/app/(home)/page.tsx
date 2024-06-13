'use client'

import { useState } from 'react'
import Image from 'next/image'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import tabbyUrl from '@/assets/tabby.png'
import { noop } from 'lodash-es'
import { useTheme } from 'next-themes'

import { SESSION_STORAGE_KEY } from '@/lib/constants'
import { useEnableSearch } from '@/lib/experiment-flags'
import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { useSignOut } from '@/lib/tabby/auth'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { CardContent, CardFooter } from '@/components/ui/card'
import { IconJetBrains, IconRotate, IconVSCode } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
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
import { UserAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import Stats from './components/stats'

const resetUserAuthTokenDocument = graphql(/* GraphQL */ `
  mutation ResetUserAuthToken {
    resetUserAuthToken
  }
`)

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

function MainPanel() {
  const [searchFlag] = useEnableSearch()
  const { data: healthInfo } = useHealth()
  const [{ data }] = useMe()
  const isChatEnabled = useIsChatEnabled()
  const signOut = useSignOut()
  const [signOutLoading, setSignOutLoading] = useState(false)
  const [isSearchOpen, setIsSearchOpen] = useState(false)
  const { theme } = useTheme()
  const [isShowDemoBanner] = useShowDemoBanner()

  if (!healthInfo || !data?.me) return <></>

  const handleSignOut = async () => {
    if (signOutLoading) return
    setSignOutLoading(true)
    await signOut()
    setSignOutLoading(false)
  }

  const onSearch = (question: string) => {
    sessionStorage.setItem(SESSION_STORAGE_KEY.SEARCH_INITIAL_MSG, question)
    window.open('/search')
    setIsSearchOpen(false)
  }

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }
  return (
    <div className="flex flex-col transition-all" style={style}>
      <header className="flex items-center justify-between px-10 py-5">
        <div />
        <div className="flex items-center gap-x-3">
          <ClientOnly>
            <ThemeToggle />
          </ClientOnly>
          <UserPanel>
            <UserAvatar className="h-10 w-10 border" />
          </UserPanel>
        </div>
      </header>

      <main className="flex flex-1 flex-col items-center justify-center">
        <div className="mx-auto -mt-[2vh] flex w-full max-w-4xl flex-col items-center">
          <Image src={tabbyUrl} alt="logo" width={45} />
          <p className="mb-6 scroll-m-20 text-xl font-semibold tracking-tight text-secondary-foreground">
            The Private Search Assistant
          </p>
          <TextAreaSearch onSearch={onSearch} isExpanded />
          <div className="mt-10 flex w-full gap-x-5">
            <div
              className="w-[21rem] rounded p-4"
              style={{ background: theme === 'dark' ? '#423929' : '#e8e1d3' }}
            >
              <Configuration />
            </div>
            <Stats />
          </div>
        </div>
      </main>
    </div>
    // <div className="lg:mt-[10vh]">
    //   <div className="mx-auto flex w-screen flex-col px-5 py-20 lg:w-auto lg:flex-row lg:justify-center lg:gap-x-10 lg:px-0 lg:py-10">
    //     <div className="relative mb-5 flex flex-col rounded-lg pb-4 lg:mb-0 lg:mt-12 lg:w-64">
    //       <UserAvatar className="h-20 w-20 border-4 border-background" />

    //       <div className="mt-2 flex w-full">
    //         <div className="flex flex-col gap-y-1">
    //           {data.me.name && (
    //             <div className="flex items-center gap-2">
    //               <IconUser className="text-muted-foreground" />
    //               <p className="max-w-[10rem] truncate text-sm">
    //                 {data.me.name}
    //               </p>
    //             </div>
    //           )}
    //           <div className="flex items-center gap-2">
    //             <IconMail className="text-muted-foreground" />
    //             <p className="max-w-[10rem] truncate text-sm">
    //               {data.me.email}
    //             </p>
    //           </div>
    //         </div>

    //         <ThemeToggle className="-mt-2 ml-auto" />
    //       </div>

    //       <Separator className="my-4" />
    //       <Configuration />

    //       <div className="mt-auto flex flex-col gap-1 lg:mb-[28px]">
    //         <MenuLink href="/profile" icon={<IconGear />}>
    //           Settings
    //         </MenuLink>
    //         {isChatEnabled && (
    //           <MenuLink href="/playground" icon={<IconChat />} target="_blank">
    //             Chat Playground
    //           </MenuLink>
    //         )}
    //         <MenuLink href="/files" icon={<IconCode />} target="_blank">
    //           Code Browser
    //         </MenuLink>
    //         {searchFlag.value && isChatEnabled && (
    //           <Dialog open={isSearchOpen} onOpenChange={setIsSearchOpen}>
    //             <DialogTrigger>
    //               <div className="flex items-center gap-2">
    //                 <div className="text-muted-foreground">
    //                   <IconSearch />
    //                 </div>
    //                 <div className="flex cursor-pointer items-center gap-1 text-sm transition-opacity hover:opacity-50">
    //                   Search
    //                 </div>
    //               </div>
    //             </DialogTrigger>
    //             <DialogContent className="dialog-without-close-btn -mt-48 border-none bg-transparent shadow-none sm:max-w-xl">
    //               <div className="flex flex-col items-center">
    //                 <Image src={logoUrl} alt="logo" width={42} />
    //                 <h4 className="mb-6 scroll-m-20 text-xl font-semibold tracking-tight text-background dark:text-foreground">
    //                   The Private Search Assistant
    //                 </h4>
    //                 <TextAreaSearch onSearch={onSearch} />
    //               </div>
    //             </DialogContent>
    //           </Dialog>
    //         )}
    //         <MenuLink icon={<IconLogout />} onClick={handleSignOut}>
    //           <span>Sign out</span>
    //           {signOutLoading && <IconSpinner className="ml-1" />}
    //         </MenuLink>
    //       </div>
    //     </div>

    //     <div className="lg:min-h-[700px] lg:w-[calc(100vw-30rem)] xl:w-[62rem]">
    //       <Stats />
    //     </div>
    //   </div>
    // </div>
  )
}

function MenuLink({
  children,
  icon,
  href,
  target,
  onClick
}: {
  children: React.ReactNode
  icon: React.ReactNode
  href?: string
  target?: string
  onClick?: () => void
}) {
  const router = useRouter()

  const onClickMenu = (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    e.stopPropagation()
    if (onClick) return onClick()
    if (href) {
      if (target === '_blank') return window.open(href)
      router.push(href)
    }
  }

  return (
    <div className="flex items-center gap-2">
      <div className="text-muted-foreground">{icon}</div>
      <div
        className="flex cursor-pointer items-center gap-1 text-sm transition-opacity hover:opacity-50"
        onClick={onClickMenu}
      >
        {children}
      </div>
    </div>
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
