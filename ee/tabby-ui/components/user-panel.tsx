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
      <DropdownMenuContent>
        <DropdownMenuLabel>{user.email}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {isChatEnabled && (
          <DropdownMenuItem>
            <Link
              target="_blank"
              href="/playground"
              className="flex w-full items-center"
            >
              <IconChat />
              <span className="ml-2">Chat Playground</span>
            </Link>
          </DropdownMenuItem>
        )}
        <DropdownMenuItem>
          <Link
            target="_blank"
            href="/files"
            className="flex w-full items-center"
          >
            <IconCode />
            <span className="ml-2">Code Browser</span>
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem>
          <Link
            target="_blank"
            href="/api"
            className="flex w-full items-center"
          >
            <IconBackpack />
            <span className="ml-2">API Docs</span>
          </Link>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={signOut}>
          <span className="flex items-center">
            <IconLogout />
            <span className="ml-2">Logout</span>
          </span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
