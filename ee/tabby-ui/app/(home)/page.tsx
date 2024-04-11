'use client'

import { useState, useTransition } from 'react'
import Link from 'next/link'
import { noop } from 'lodash-es'
import { useTheme } from 'next-themes'

import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useSignOut } from '@/lib/tabby/auth'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { CardContent, CardFooter } from '@/components/ui/card'
import {
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
    <div className="flex flex-1 justify-center">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-x-5 px-5 py-20 md:w-auto md:flex-row md:py-10 lg:gap-x-10 xl:px-0">
        <div className="relative mb-5 flex flex-col rounded-lg pb-4 lg:mb-0 lg:mt-12 lg:w-64">
          <span>
            <UserPanel>
              <UserAvatar className="h-20 w-20 border-4 border-background" />
            </UserPanel>
          </span>
          <div className="mt-2 flex w-full">
            <div className="flex items-center gap-2">
              <IconMail className="text-muted-foreground" />
              <p className="max-w-[10rem] truncate text-sm">{data.me.email}</p>
            </div>

            <ThemeToggle className="ml-auto" />
          </div>

          <Configuration />

          <Separator />

          <div className="mt-[48px] flex flex-col gap-1">
            <Link
              className="flex items-center gap-2 text-sm transition-opacity hover:opacity-50"
              href="/profile"
            >
              <IconGear className="text-muted-foreground" />
              Settings
            </Link>

            <div
              className="flex cursor-pointer items-center gap-2 py-1"
              onClick={handleSignOut}
            >
              <IconLogout className="text-muted-foreground" />
              <span className="text-sm transition-opacity hover:opacity-50">
                Sign out
              </span>
              {signOutLoading && <IconSpinner className="ml-1" />}
            </div>
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
