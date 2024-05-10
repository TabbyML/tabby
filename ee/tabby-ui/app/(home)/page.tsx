'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { noop } from 'lodash-es'

import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { useSignOut } from '@/lib/tabby/auth'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { CardContent, CardFooter } from '@/components/ui/card'
import {
  IconChat,
  IconCode,
  IconGear,
  IconLogout,
  IconMail,
  IconRotate,
  IconSpinner
} from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { CopyButton } from '@/components/copy-button'
import SlackDialog from '@/components/slack-dialog'
import { ThemeToggle } from '@/components/theme-toggle'
import { UserAvatar } from '@/components/user-avatar'

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
    </div>
  )
}

function MainPanel() {
  const { data: healthInfo } = useHealth()
  const [{ data }] = useMe()
  const isChatEnabled = useIsChatEnabled()
  const signOut = useSignOut()
  const [signOutLoading, setSignOutLoading] = useState(false)

  if (!healthInfo || !data?.me) return <></>

  const handleSignOut = async () => {
    if (signOutLoading) return
    setSignOutLoading(true)
    await signOut()
    setSignOutLoading(false)
  }

  return (
    <div className="flex flex-1 justify-center lg:mt-[10vh]">
      <div className="mx-auto flex w-screen flex-col px-5 py-20 lg:w-auto lg:flex-row lg:justify-center lg:gap-x-10 lg:px-0 lg:py-10">
        <div className="relative mb-5 flex flex-col rounded-lg pb-4 lg:mb-0 lg:mt-12 lg:w-64">
          <UserAvatar className="h-20 w-20 border-4 border-background" />

          <div className="mt-2 flex w-full">
            <div className="flex items-center gap-2">
              <IconMail className="text-muted-foreground" />
              <p className="max-w-[10rem] truncate text-sm">{data.me.email}</p>
            </div>

            <ThemeToggle className="ml-auto" />
          </div>

          <Separator className="my-4" />
          <Configuration />

          <div className="mt-auto flex flex-col gap-1 lg:mb-[28px]">
            <MenuLink href="/profile" icon={<IconGear />}>
              Settings
            </MenuLink>
            {isChatEnabled && (
              <MenuLink href="/playground" icon={<IconChat />} target="_blank">
                Chat Playground
              </MenuLink>
            )}
            <MenuLink href="/files" icon={<IconCode />} target="_blank">
              Code Browser
            </MenuLink>
            <MenuLink icon={<IconLogout />} onClick={handleSignOut}>
              <span>Sign out</span>
              {signOutLoading && <IconSpinner className="ml-1" />}
            </MenuLink>
          </div>
        </div>

        <div className="lg:min-h-[700px] lg:w-[calc(100vw-30rem)] xl:w-[62rem]">
          <Stats />
        </div>
      </div>
    </div>
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
