'use client'

import { useContext, useState } from 'react'
import type { MouseEvent } from 'react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'

import { clearHomeScrollPosition } from '@/lib/stores/scroll-store'
import { useMutation } from '@/lib/tabby/gql'
import { deleteThreadMutation } from '@/lib/tabby/query'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle
} from '@/components/ui/alert-dialog'
import { Badge } from '@/components/ui/badge'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import {
  IconBookOpen,
  IconChevronLeft,
  IconMore,
  IconPlus,
  IconShare,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import { ClientOnly } from '@/components/client-only'
import { NotificationBox } from '@/components/notification-box'
import { ThemeToggle } from '@/components/theme-toggle'
import { MyAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import { SearchContext } from './search-context'

type HeaderProps = {
  threadIdFromURL?: string
  streamingDone?: boolean
  threadId?: string
  onConvertToPage?: () => void
  onShare?: () => void
}

export function Header({
  threadIdFromURL,
  streamingDone,
  threadId,
  onConvertToPage,
  onShare
}: HeaderProps) {
  const router = useRouter()
  const { isThreadOwner } = useContext(SearchContext)
  const [deleteAlertVisible, setDeleteAlertVisible] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const deleteThread = useMutation(deleteThreadMutation, {
    onCompleted(data) {
      if (data.deleteThread) {
        router.replace('/')
      } else {
        toast.error('Failed to delete')
        setIsDeleting(false)
      }
    },
    onError(err) {
      toast.error(err?.message || 'Failed to delete')
      setIsDeleting(false)
    }
  })

  const handleDeleteThread = (e: MouseEvent<HTMLButtonElement>) => {
    if (!threadId) return

    e.preventDefault()
    setIsDeleting(true)
    deleteThread({
      id: threadId
    })
  }

  const onNavigateToHomePage = (scroll?: boolean) => {
    if (scroll) {
      clearHomeScrollPosition()
    }
    router.push('/')
  }

  return (
    <header className="flex h-16 items-center justify-between px-4 lg:px-10">
      <div className="flex items-center gap-x-6">
        <Button
          variant="ghost"
          className="-ml-1 pl-0 text-sm text-muted-foreground"
          onClick={() => onNavigateToHomePage()}
        >
          <IconChevronLeft className="mr-1 h-5 w-5" />
          Home
        </Button>
      </div>
      <div className="flex items-center gap-2">
        {streamingDone && threadIdFromURL && (
          <Button
            variant="ghost"
            className="flex items-center gap-1 px-2 font-normal"
            onClick={() => onNavigateToHomePage(true)}
          >
            <IconPlus />
          </Button>
        )}
        {streamingDone && threadId && (
          <DropdownMenu modal={false}>
            <DropdownMenuTrigger asChild>
              <Button size="icon" variant="ghost">
                <IconMore />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {!!onShare && (
                <DropdownMenuItem
                  className="cursor-pointer gap-2"
                  onSelect={e => onShare()}
                >
                  <IconShare />
                  Share
                </DropdownMenuItem>
              )}
              {isThreadOwner && (
                <>
                  <DropdownMenuItem
                    className="cursor-pointer gap-2"
                    onSelect={e => onConvertToPage?.()}
                  >
                    <IconBookOpen />
                    <span>Convert to page</span>
                    <Badge
                      variant="outline"
                      className="h-3.5 border-secondary-foreground/60 px-1.5 text-[10px] text-secondary-foreground/60"
                    >
                      Beta
                    </Badge>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className="cursor-pointer gap-2 !text-destructive"
                    onSelect={e => setDeleteAlertVisible(true)}
                  >
                    <IconTrash />
                    Delete
                  </DropdownMenuItem>
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
        <ClientOnly>
          <ThemeToggle />
        </ClientOnly>
        <NotificationBox className="mr-4" />
        <UserPanel
          showHome={false}
          showSetting
          beforeRouteChange={() => {
            clearHomeScrollPosition()
          }}
        >
          <MyAvatar className="h-10 w-10 border" />
        </UserPanel>
      </div>
      <AlertDialog
        open={deleteAlertVisible}
        onOpenChange={setDeleteAlertVisible}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this thread</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this thread? This operation is not
              revertible.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className={buttonVariants({ variant: 'destructive' })}
              onClick={handleDeleteThread}
            >
              {isDeleting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Yes, delete it
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </header>
  )
}
