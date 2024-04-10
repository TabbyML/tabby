import React from 'react'

import { useMe } from '@/lib/hooks/use-me'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { useSignOut } from '@/lib/tabby/auth'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { UserAvatar } from '@/components/user-avatar'

import {
  IconBackpack,
  IconChat,
  IconCode,
  IconLogout,
  IconSpinner,
  IconHome
} from './ui/icons'

export default function UserPanel({
  trigger,
  align
}: {
  trigger?: React.ReactNode
  align?: 'start' | 'center' | 'end'
}) {
  const signOut = useSignOut()
  const [{ data }] = useMe()
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
      <DropdownMenuTrigger>
        {trigger || <UserAvatar className="h-10 w-10 border" />}
      </DropdownMenuTrigger>
      <DropdownMenuContent
        collisionPadding={{ right: 16 }}
        align={align || 'center'}
      >
        <DropdownMenuLabel>{user.email}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={() => window.open('/')}
          className="cursor-pointer"
        >
          <IconHome />
          <span className="ml-2">Home</span>
        </DropdownMenuItem>
        {isChatEnabled && (
          <DropdownMenuItem
            onClick={() => window.open('/playground')}
            className="cursor-pointer"
          >
            <IconChat />
            <span className="ml-2">Chat Playground</span>
          </DropdownMenuItem>
        )}
        <DropdownMenuItem
          onClick={() => window.open('/files')}
          className="cursor-pointer"
        >
          <IconCode />
          <span className="ml-2">Code Browser</span>
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={() => window.open('/api')}
          className="cursor-pointer"
        >
          <IconBackpack />
          <span className="ml-2">API Docs</span>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          disabled={signOutLoading}
          onClick={handleSignOut}
          className="cursor-pointer"
        >
          <IconLogout />
          <span className="ml-2">Logout</span>
          {signOutLoading && <IconSpinner className="ml-1" />}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
