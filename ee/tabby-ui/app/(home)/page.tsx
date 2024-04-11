'use client'

import { useState, useTransition } from 'react'
import Link from 'next/link'
import { noop, capitalize } from 'lodash-es'
import { useTheme } from 'next-themes'

import { useSignOut } from '@/lib/tabby/auth'
import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { CardContent, CardFooter } from '@/components/ui/card'
import {
  IconChevronRight,
  IconMail,
  IconRotate,
  IconGear,
  IconLogout,
  IconSpinner,
  IconMoon, IconSun
} from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { CopyButton } from '@/components/copy-button'
import SlackDialog from '@/components/slack-dialog'
import { UserAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import Stats from './components/stats'

const resetUserAuthTokenDocument = graphql(/* GraphQL */ `
  mutation ResetUserAuthToken {
    resetUserAuthToken
  }
`)

function Configuration() {
  const [{ data }, reexecuteQuery] = useMe()
  const externalUrl = useExternalURL()

  const resetUserAuthToken = useMutation(resetUserAuthTokenDocument, {
    onCompleted: () => reexecuteQuery()
  })

  if (!data?.me) return <></>

  return (
    <div className="mb-1 mt-6">
      <CardContent className="flex flex-col gap-6 px-0">
        <div className="flex flex-col">
          <Label className="text-xs font-semibold">Endpoint URL</Label>
          <span className="flex items-center gap-1">
            <Input
              value={externalUrl}
              onChange={noop}
              className="h-7 max-w-[320px] rounded-none border-x-0 !border-t-0 border-muted-foreground p-0 shadow-none dark:border-primary/50"
            />
            <CopyButton value={externalUrl} />
          </span>
        </div>

        <div className="flex flex-col">
          <Label className="text-xs font-semibold">Token</Label>
          <span className="flex items-center gap-1">
            <Input
              className="h-7 max-w-[320px] rounded-none border-x-0 !border-t-0 border-muted-foreground p-0 font-mono shadow-none dark:border-primary/50"
              value={data.me.authToken}
              onChange={noop}
            />
            <Button
              title="Rotate"
              size="icon"
              variant="hover-destructive"
              onClick={() => resetUserAuthToken()}
            >
              <IconRotate />
            </Button>
            <CopyButton value={data.me.authToken} />
          </span>
        </div>
      </CardContent>
      <CardFooter className="px-0 text-xs text-muted-foreground">
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
  const [signOutLoading, setSignOutLoading] = useState(false)
  const { setTheme, theme } = useTheme()
  const [_, startTransition] = useTransition()
  const signOut = useSignOut()
  const { data: healthInfo } = useHealth()
  const [{ data }] = useMe()

  if (!healthInfo || !data?.me) return <></>

  const handleSignOut = async () => {
    if (signOutLoading) return

    setSignOutLoading(true)
    await signOut()
    setSignOutLoading(false)
  }

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="mx-auto flex w-screen max-w-7xl flex-col gap-x-5 px-5 py-20 md:w-auto md:flex-row md:py-10 lg:gap-x-10 xl:px-0">
        <div className="relative mb-5 flex flex-col rounded-lg px-5 pb-4 lg:mb-0 lg:w-64">
          <UserPanel
            trigger={
              <UserAvatar className="h-20 w-20 border-4 border-background" />
            }
            align="start"
          />
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger className="mt-2 w-full cursor-default">
                <div className="flex items-center">
                  <IconMail className="mr-2 text-muted-foreground" />
                  <p className="max-w-[10rem] truncate text-sm">
                    {data.me.email}
                  </p>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>{data.me.email}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
    
          <Configuration />

          <div className="flex items-center py-1">
            <IconGear className="mr-2 text-muted-foreground" />
            <Link
              className="flex items-center gap-x-1 text-sm transition-opacity hover:opacity-50"
              href="/profile"
            >
              <span>Settings</span>
              <IconChevronRight />
            </Link>
          </div>
          <div
            className="flex cursor-pointer items-center py-1"
            onClick={() => {
              startTransition(() => {
                setTheme(theme === 'light' ? 'dark' : 'light')
              })
            }}
          >
            {theme === 'dark' ? (
              <IconMoon className="mr-2 text-muted-foreground transition-all" />
            ) : (
              <IconSun className="mr-2 text-muted-foreground transition-all" />
            )}
            <span className="text-sm transition-opacity hover:opacity-50">{capitalize(theme)}</span>
          </div>

          <div className="flex cursor-pointer items-center py-1" onClick={handleSignOut}>
            <IconLogout className="mr-2 text-muted-foreground" />
            <span className="text-sm transition-opacity hover:opacity-50">Logout</span>
            {signOutLoading && <IconSpinner className="ml-1" />}
          </div>
        </div>

        <Stats />
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
