import React from 'react'
import { useRouter } from 'next/navigation'

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

import {
  IconBackpack,
  IconCode,
  IconGear,
  IconHome,
  IconLogout,
  IconSpinner
} from './ui/icons'

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
      <DropdownMenuTrigger>{children}</DropdownMenuTrigger>
      <DropdownMenuContent collisionPadding={{ right: 16 }}>
        {user.name && (
          <>
            <DropdownMenuLabel className="pb-0.5">
              {user.name}
            </DropdownMenuLabel>
            <DropdownMenuLabel className="pb-1 pt-0 text-sm font-normal text-muted-foreground">
              {user.email}
            </DropdownMenuLabel>
          </>
        )}
        {!user.name && <DropdownMenuLabel>{user.email}</DropdownMenuLabel>}
        <DropdownMenuSeparator />
        {showHome && (
          <DropdownMenuItem
            onClick={() => router.push('/')}
            className="cursor-pointer"
          >
            <IconHome />
            <span className="ml-2">Home</span>
          </DropdownMenuItem>
        )}
        {showSetting && (
          <DropdownMenuItem
            onClick={() => router.push('/profile')}
            className="cursor-pointer"
          >
            <IconGear />
            <span className="ml-2">Settings</span>
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
          <span className="ml-2">Sign out</span>
          {signOutLoading && <IconSpinner className="ml-1" />}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
