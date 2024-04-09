'use client'

import { Dispatch, SetStateAction, useState } from 'react'
import Link from 'next/link'
import { noop } from 'lodash-es'

import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'

import { Badge, badgeVariants } from '@/components/ui/badge'
import { Button, buttonVariants } from '@/components/ui/button'
import { CardContent, CardFooter } from '@/components/ui/card'
import { IconMoveRight, IconRotate, IconSettings } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { CopyButton } from '@/components/copy-button'
import SlackDialog from '@/components/slack-dialog'
import { ThemeToggle } from '@/components/theme-toggle'
import { UserAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'
import Profile from './components/profile'
import Stats from './components/stats'

const resetUserAuthTokenDocument = graphql(/* GraphQL */ `
  mutation ResetUserAuthToken {
    resetUserAuthToken
  }
`)

enum Menu {
  Stats = 'stats',
  Config = 'config',
  Profile = 'profile'
}

function MenuItem({
  value,
  label,
  current,
  setMenu
}: {
  value: Menu
  label: string
  current: Menu
  setMenu: Dispatch<SetStateAction<Menu>>
}) {
  return (
    <Badge
      onClick={setMenu.bind(null, value)}
      variant={current === value ? 'default' : 'outline'}
      className="cursor-pointer"
    >
      {label}
    </Badge>
  )
}

function Configuration() {
  const [{ data }, reexecuteQuery] = useMe()
  const externalUrl = useExternalURL()

  const resetUserAuthToken = useMutation(resetUserAuthTokenDocument, {
    onCompleted: () => reexecuteQuery()
  })

  if (!data?.me) return <></>

  return (
    <div className="py-1">
      <CardContent className="flex flex-col gap-6 px-0">
        <div className="flex flex-col gap-2">
          <Label className="font-semibold">Endpoint URL</Label>
          <span className="flex items-center gap-1">
            <Input
              value={externalUrl}
              onChange={noop}
              className="max-w-[320px]"
            />
            <CopyButton value={externalUrl} />
          </span>
        </div>

        <div className="flex flex-col gap-2">
          <Label className="font-semibold">Token</Label>
          <span className="flex items-center gap-1">
            <Input
              className="max-w-[320px] font-mono"
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
      <CardFooter className="px-0 text-sm">
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
  const [menu, setMenu] = useState<Menu>(Menu.Stats)
  const { data: healthInfo } = useHealth()
  const [{ data }] = useMe()

  if (!healthInfo || !data?.me) return <></>
  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-y-5 px-10 pb-20 pt-10 md:pt-40 lg:px-0">
      <div className="flex justify-between">
        <div>
          <UserAvatar className="relative h-20 w-20 border" />
          <p className="mt-1.5">{data.me.email}</p>
        </div>

        <div className="flex items-center gap-x-2 self-start">
          <ThemeToggle />
          <UserPanel
            trigger={
              <div
                className={cn(
                  buttonVariants({ variant: 'ghost' }),
                  'flex items-center justify-center px-2'
                )}
              >
                <IconSettings />
              </div>
            }
            align="end"
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <MenuItem
          value={Menu.Stats}
          label="Stats"
          current={menu}
          setMenu={setMenu}
        />
        <MenuItem
          value={Menu.Config}
          label="Configuration"
          current={menu}
          setMenu={setMenu}
        />
        <MenuItem
          value={Menu.Profile}
          label="Profile"
          current={menu}
          setMenu={setMenu}
        />
        {data.me.isAdmin && (
          <Link
            className={cn(
              badgeVariants({ variant: 'outline' }),
              'flex items-center gap-x-2'
            )}
            href="/cluster"
          >
            <span>Admin Dashboard</span>
            <IconMoveRight />
          </Link>
        )}
      </div>

      {menu === Menu.Stats && <Stats />}
      {menu === Menu.Config && <Configuration />}
      {menu === Menu.Profile && <Profile />}
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
