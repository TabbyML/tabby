import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { noop } from 'lodash-es'
import { UseQueryExecute } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { MeQueryQuery } from '@/lib/gql/generates/graphql'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { useSignOut } from '@/lib/tabby/auth'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { CardContent, CardFooter } from '@/components/ui/card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { IconJetBrains, IconRotate, IconVSCode } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { CopyButton } from '@/components/copy-button'

import {
  IconBackpack,
  IconCode,
  IconGear,
  IconHome,
  IconLogout,
  IconSpinner
} from './ui/icons'
import { MyAvatar, UserAvatar } from './user-avatar'

const resetUserAuthTokenDocument = graphql(/* GraphQL */ `
  mutation ResetUserAuthToken {
    resetUserAuthToken
  }
`)

export default function UserPanel({
  children,
  showHome = true,
  showSetting = false
}: {
  children?: React.ReactNode
  showHome?: boolean
  showSetting?: boolean
}) {
  const router = useRouter()
  const signOut = useSignOut()
  const [{ data }, reexecuteQuery] = useMe()
  const user = data?.me
  const isChatEnabled = useIsChatEnabled()
  const [signOutLoading, setSignOutLoading] = React.useState(false)
  const handleSignOut: React.MouseEventHandler<HTMLDivElement> = async e => {
    e.preventDefault()

    setSignOutLoading(true)
    await signOut()
    setSignOutLoading(false)
  }

  if (!user) {
    return
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger>{children}</DropdownMenuTrigger>
      <DropdownMenuContent
        side="bottom"
        align="end"
        className="p-0 overflow-y-auto"
      >
        <div className="p-4 space-y-4">
          <div className="flex gap-2 items-center">
            <UserAvatar
              user={user}
              className="w-12 h-12 border-[2px] border-white shrink-0"
            />
            <div className="space-y-1">
              {user.name && (
                <>
                  <DropdownMenuLabel className="p-0">
                    {user.name}
                  </DropdownMenuLabel>
                  <DropdownMenuLabel className="p-0 text-sm font-normal text-muted-foreground">
                    {user.email}
                  </DropdownMenuLabel>
                </>
              )}
              {!user.name && (
                <DropdownMenuLabel>{user.email}</DropdownMenuLabel>
              )}
            </div>
          </div>

          <Configuration user={user} reexecuteQuery={reexecuteQuery} />
        </div>

        <DropdownMenuSeparator className="mt-0 mb-1" />

        <div className="px-1.5">
          {showHome && (
            <DropdownMenuItem
              onClick={() => router.push('/')}
              className="cursor-pointer pl-3 py-2"
            >
              <IconHome />
              <span className="ml-2">Home</span>
            </DropdownMenuItem>
          )}
          {showSetting && (
            <DropdownMenuItem
              onClick={() => router.push('/profile')}
              className="cursor-pointer pl-3 py-2"
            >
              <IconGear />
              <span className="ml-2">Settings</span>
            </DropdownMenuItem>
          )}
          <DropdownMenuItem
            onClick={() => window.open('/files')}
            className="cursor-pointer pl-3 py-2"
          >
            <IconCode />
            <span className="ml-2">Code Browser</span>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => window.open('/api')}
            className="cursor-pointer pl-3 py-2"
          >
            <IconBackpack />
            <span className="ml-2">API Docs</span>
          </DropdownMenuItem>
        </div>

        <DropdownMenuSeparator />
        <DropdownMenuItem
          disabled={signOutLoading}
          onClick={handleSignOut}
          className="cursor-pointer pl-3 py-2 mx-1.5 mb-1.5"
        >
          <IconLogout />
          <span className="ml-2">Sign out</span>
          {signOutLoading && <IconSpinner className="ml-1" />}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function Configuration({
  className,
  user,
  reexecuteQuery
}: {
  user: MeQueryQuery['me']
  reexecuteQuery: UseQueryExecute
  className?: string
}) {
  const externalUrl = useExternalURL()

  const resetUserAuthToken = useMutation(resetUserAuthTokenDocument, {
    onCompleted: () => reexecuteQuery()
  })

  return (
    <>
      <div
        className={cn(
          'bg-[#FBF5ED] dark:bg-[#3D382F] p-4 w-[268px] rounded-xl',
          className
        )}
      >
        <CardContent className="p-0">
          <div className="flex flex-col gap-2">
            <Label className="text-xs text-muted-foreground">
              Endpoint URL
            </Label>
            <span className="group relative">
              <Input
                value={externalUrl}
                onChange={noop}
                className="bg-[#FEFCF8] dark:bg-[#4F483B] border-none group-hover:pr-12 group-focus-within:pr-12"
              />
              <CopyButton
                value={externalUrl}
                className={cn(
                  'absolute right-1 top-0.5 hidden group-hover:flex group-focus-within:flex'
                )}
              />
            </span>
          </div>

          <div className="flex flex-col gap-2 mt-4">
            <Label className="text-xs text-muted-foreground">Token</Label>
            <span className="group relative">
              <Input
                value={user.authToken}
                onChange={noop}
                className="bg-[#FEFCF8] dark:bg-[#4F483B] border-none group-hover:pr-20 group-focus-within:pr-20"
              />
              <div className="absolute right-1 top-0.5 hidden group-hover:flex group-focus-within:flex items-center gap-1">
                <CopyButton value={user.authToken} />
                <Button
                  title="Rotate"
                  size="icon"
                  variant="hover-destructive"
                  onClick={() => resetUserAuthToken()}
                >
                  <IconRotate />
                </Button>
              </div>
            </span>
          </div>
        </CardContent>
        <CardFooter className="p-0 mt-3 text-xs text-muted-foreground">
          <span>
            Use information above for IDE extensions / plugins configuration,
            see{' '}
            <a
              className="underline text-link"
              target="_blank"
              href="https://tabby.tabbyml.com/docs/extensions/configurations#server"
            >
              documentation website
            </a>{' '}
            for details
          </span>
        </CardFooter>
      </div>
      <div className="mb-6 mt-3 flex gap-x-4 lg:mb-0">
        <IDELink
          href="https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby"
          name="Visual Studio Code"
          icon={<IconVSCode className="h-6 w-6" />}
        />
        <IDELink
          href="https://plugins.jetbrains.com/plugin/22379-tabby"
          name="JetBrains"
          icon={<IconJetBrains className="h-6 w-6" />}
        />
      </div>
    </>
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
          className="bg-[#FBF5ED] dark:bg-[#3D382F] w-8 h-8 flex items-center justify-center rounded-lg text-[#030302]"
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
