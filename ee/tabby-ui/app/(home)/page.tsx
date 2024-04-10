'use client'

import Link from 'next/link'
import { noop } from 'lodash-es'

import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { CardContent, CardFooter } from '@/components/ui/card'
import {
  IconBarChart,
  IconChevronRight,
  IconMail,
  IconRotate,
  IconUser
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
    <div className="mt-6">
      <CardContent className="flex flex-col gap-6 px-0">
        <div className="flex flex-col">
          <Label className="text-xs font-semibold">Endpoint URL</Label>
          <span className="flex items-center gap-1">
            <Input
              value={externalUrl}
              onChange={noop}
              className="h-7 max-w-[320px] rounded-none border-x-0 !border-t-0 border-primary-foreground/50 p-0 shadow-none dark:border-primary/50"
            />
            <CopyButton value={externalUrl} />
          </span>
        </div>

        <div className="flex flex-col">
          <Label className="text-xs font-semibold">Token</Label>
          <span className="flex items-center gap-1">
            <Input
              className="h-7 max-w-[320px] rounded-none border-x-0 !border-t-0 border-primary-foreground/50 p-0 font-mono shadow-none dark:border-primary/50"
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
      <CardFooter className="px-0 text-xs text-primary-foreground/50 dark:text-primary/50">
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

  if (!healthInfo || !data?.me) return <></>
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="mx-auto flex w-screen max-w-7xl flex-col gap-x-5 px-5 py-20 md:w-auto  md:flex-row md:py-10 lg:gap-x-10 xl:px-0">
        <div className="relative mb-5 flex flex-col rounded-lg bg-primary/90 px-5 text-primary-foreground dark:bg-primary-foreground/90 dark:text-primary lg:mb-0 lg:w-64">
          <div className="absolute right-0 top-0">
            <ThemeToggle />
          </div>
          <UserPanel
            trigger={
              <UserAvatar className="-mt-10 h-20 w-20 border-4 border-background" />
            }
            align="start"
          />
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger className="mt-2 w-full cursor-default">
                <div className="flex items-center">
                  <IconMail className="mr-2 text-primary-foreground/50 dark:text-primary/50" />
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
          <div className="flex items-center py-1">
            <IconUser className="mr-2 text-primary-foreground/50 dark:text-primary/50" />
            <Link
              className="flex items-center gap-x-1 text-sm transition-opacity hover:opacity-50"
              href="/profile"
            >
              <span>Profile</span>
              <IconChevronRight />
            </Link>
          </div>
          {data.me.isAdmin && (
            <div className="flex items-center py-1">
              <IconBarChart className="mr-2 text-primary-foreground/50 dark:text-primary/50" />
              <Link
                className="flex items-center gap-x-1 text-sm transition-opacity hover:opacity-50"
                href="/cluster"
              >
                <span>Admin Dashboard</span>
                <IconChevronRight />
              </Link>
            </div>
          )}
          <Configuration />
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
