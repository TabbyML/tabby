import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { noop } from 'lodash-es'
import { UseQueryExecute } from 'urql'

import { useEnablePage } from '@/lib/experiment-flags'
import { graphql } from '@/lib/gql/generates'
import { MeQueryQuery } from '@/lib/gql/generates/graphql'
import { useMe } from '@/lib/hooks/use-me'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
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
import {
  IconBookOpen,
  IconJetBrains,
  IconRotate,
  IconVSCode
} from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { CopyButton } from '@/components/copy-button'

import { Badge } from './ui/badge'
import {
  IconBackpack,
  IconCode,
  IconGear,
  IconHome,
  IconLogout,
  IconSpinner
} from './ui/icons'
import { UserAvatar } from './user-avatar'

const resetUserAuthTokenDocument = graphql(/* GraphQL */ `
  mutation ResetUserAuthToken {
    resetUserAuthToken
  }
`)

export default function UserPanel({
  children,
  showHome = true,
  showSetting = false,
  beforeRouteChange
}: {
  children?: React.ReactNode
  showHome?: boolean
  showSetting?: boolean
  beforeRouteChange?: (nextPathname?: string) => void
}) {
  const router = useRouter()
  const signOut = useSignOut()
  const [{ data }, reexecuteQuery] = useMe()
  const user = data?.me
  const [signOutLoading, setSignOutLoading] = React.useState(false)
  const handleSignOut: React.MouseEventHandler<HTMLDivElement> = async e => {
    e.preventDefault()

    setSignOutLoading(true)
    await signOut()
    setSignOutLoading(false)
  }
  const [enablePage] = useEnablePage()

  const onNavigate = (pathname: string, replace?: boolean) => {
    beforeRouteChange?.(pathname)
    if (replace) {
      router.replace(pathname)
    } else {
      router.push(pathname)
    }
  }

  if (!user) {
    return
  }

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger>{children}</DropdownMenuTrigger>
      <DropdownMenuContent
        side="bottom"
        align="end"
        className="relative overflow-y-auto p-0"
        style={{ maxHeight: 'calc(100vh - 6rem)' }}
      >
        <div className="p-4 pt-0">
          <div className="sticky top-0 z-10 flex items-center gap-2 bg-popover pb-2 pt-4">
            <UserAvatar
              user={user}
              className="h-12 w-12 shrink-0 border-[2px] border-white"
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

          <Configuration
            className="mt-2"
            user={user}
            reexecuteQuery={reexecuteQuery}
          />
        </div>

        <DropdownMenuSeparator className="mb-1 mt-0" />

        <div className="px-1.5">
          {showHome && (
            <DropdownMenuItem
              onClick={() => onNavigate('/')}
              className="cursor-pointer py-2 pl-3"
            >
              <IconHome />
              <span className="ml-2">Home</span>
            </DropdownMenuItem>
          )}
          {!!enablePage.value && (
            <DropdownMenuItem
              onClick={() => onNavigate('/pages/new')}
              className="cursor-pointer py-2 pl-3"
            >
              <IconBookOpen />
              <span className="ml-2">New Page</span>
              <Badge
                variant="outline"
                className="ml-2 h-3.5 border-secondary-foreground/60 px-1.5 text-[10px] text-muted-foreground"
              >
                Beta
              </Badge>
            </DropdownMenuItem>
          )}
          <DropdownMenuItem
            onClick={() => window.open('/files')}
            className="cursor-pointer py-2 pl-3"
          >
            <IconCode />
            <span className="ml-2">Code Browser</span>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => window.open('/api')}
            className="cursor-pointer py-2 pl-3"
          >
            <IconBackpack />
            <span className="ml-2">API Docs</span>
          </DropdownMenuItem>
          {showSetting && (
            <DropdownMenuItem
              onClick={() => onNavigate('/profile')}
              className="cursor-pointer py-2 pl-3"
            >
              <IconGear />
              <span className="ml-2">Settings</span>
            </DropdownMenuItem>
          )}
        </div>

        <DropdownMenuSeparator />
        <DropdownMenuItem
          disabled={signOutLoading}
          onClick={handleSignOut}
          className="mx-1.5 mb-1.5 cursor-pointer py-2 pl-3"
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
          'w-[268px] rounded-xl bg-[#FBF5ED] p-4 dark:bg-[#3D382F]',
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
                className="border-none bg-[#FEFCF8] group-focus-within:pr-12 group-hover:pr-12 dark:bg-[#4F483B]"
              />
              <CopyButton
                value={externalUrl}
                className={cn(
                  'absolute right-1 top-0.5 hidden group-focus-within:flex group-hover:flex'
                )}
              />
            </span>
          </div>

          <div className="mt-4 flex flex-col gap-2">
            <Label className="text-xs text-muted-foreground">Token</Label>
            <span className="group relative">
              <Input
                value={user.authToken}
                onChange={noop}
                className="border-none bg-[#FEFCF8] group-focus-within:pr-20 group-hover:pr-20 dark:bg-[#4F483B]"
              />
              <div className="absolute right-1 top-0.5 hidden items-center gap-1 group-focus-within:flex group-hover:flex">
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
        <CardFooter className="mt-3 p-0 text-xs text-muted-foreground">
          <span>
            Use information above for IDE extensions / plugins configuration,
            see{' '}
            <a
              className="text-link underline"
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
          className="flex h-8 w-8 items-center justify-center rounded-lg text-[#030302]"
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
