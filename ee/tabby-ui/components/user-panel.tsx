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
  IconSpinner
} from './ui/icons'

export default function UserPanel() {
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
        <UserAvatar className="size-10 border" />
      </DropdownMenuTrigger>
      <DropdownMenuContent collisionPadding={{ right: 16 }}>
        <DropdownMenuLabel>{user.email}</DropdownMenuLabel>
        <DropdownMenuSeparator />
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
