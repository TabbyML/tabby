import { useAuthenticatedSession, useIsAdminInitialized, useSession, useSignOut } from '@/lib/tabby/auth'
import { cn } from '@/lib/utils'
import { IconLogout, IconUnlock } from './ui/icons'
import Link from 'next/link'
import React from 'react'

export default function UserPanel() {
  const isAdminInitialized = useIsAdminInitialized()

  const Component = isAdminInitialized ? UserInfoPanel : EnableAdminPanel;

  return <div className="py-4 flex justify-center text-sm font-medium">
    <Component className={cn('flex items-center gap-2')} />
  </div>
}

function UserInfoPanel({ className }: React.ComponentProps<'span'>) {
  const session = useAuthenticatedSession()
  const signOut = useSignOut()

  return session && <span className={className}>
    <span title="Sign out">
      <IconLogout className="cursor-pointer" onClick={signOut} />
    </span>
    {session.email}
  </span>
}

function EnableAdminPanel({ className }: React.ComponentProps<'span'>) {
  return <Link
    className={cn("cursor-pointer", className)}
    title="Authentication is currently not enabled. Click to view details"
    href={{
      pathname: "/auth/signup",
      query: { isAdmin: true }
    }}>
    <IconUnlock /> Secure Access
  </Link>
}