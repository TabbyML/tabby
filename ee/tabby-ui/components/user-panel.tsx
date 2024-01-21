import React from 'react'
import Link from 'next/link'
import { has } from 'lodash-es'
import NiceAvatar, { genConfig } from 'react-nice-avatar'

import { WorkerKind } from '@/lib/gql/generates/graphql'
import { useWorkers } from '@/lib/hooks/use-workers'
import { useAuthenticatedSession, useSignOut } from '@/lib/tabby/auth'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'

import { IconBackpack, IconChat, IconCode, IconLogout } from './ui/icons'

export default function UserPanel() {
  const user = useAuthenticatedSession()
  const signOut = useSignOut()

  const workers = useWorkers()
  const isChatEnabled = has(workers, WorkerKind.Chat)

  if (!user) {
    return
  }

  const config = genConfig(user.email)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger>
        <span className="flex h-10 w-10 rounded-full border">
          <NiceAvatar className="w-full" {...config} />
        </span>
      </DropdownMenuTrigger>
      <DropdownMenuContent collisionPadding={{ right: 16 }}>
        <DropdownMenuLabel>{user.email}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {isChatEnabled && (
          <DropdownMenuItem onClick={() => window.open("/playground")} className='cursor-pointer'>
            <IconChat />
            <span className="ml-2">Chat Playground</span>
          </DropdownMenuItem>
        )}
        <DropdownMenuItem onClick={() => window.open("/files")} className='cursor-pointer'>
          <IconCode />
          <span className="ml-2">Code Browser</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => window.open("/api")} className='cursor-pointer'>
          <IconBackpack />
          <span className="ml-2">API Docs</span>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={signOut} className='cursor-pointer'>
          <IconLogout />
          <span className="ml-2">Logout</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
