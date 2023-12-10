import React from 'react'

import { useAuthenticatedSession, useSignOut } from '@/lib/tabby/auth'

import { IconLogout } from './ui/icons'

export default function UserPanel() {
  const session = useAuthenticatedSession()
  const signOut = useSignOut()

  return (
    session && (
      <div className="flex justify-center py-4 text-sm font-medium">
        <span className="flex items-center gap-2">
          <span title="Sign out">
            <IconLogout className="cursor-pointer" onClick={signOut} />
          </span>
          {session.email}
        </span>
      </div>
    )
  )
}
