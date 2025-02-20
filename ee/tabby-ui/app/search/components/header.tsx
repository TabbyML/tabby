'use client'

import { useContext, useState } from 'react'
import type { MouseEvent } from 'react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'

import { useEnablePage } from '@/lib/experiment-flags'
import { graphql } from '@/lib/gql/generates'
import { updatePendingThreadId } from '@/lib/stores/page-store'
import { clearHomeScrollPosition } from '@/lib/stores/scroll-store'
import { useMutation } from '@/lib/tabby/gql'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger
} from '@/components/ui/alert-dialog'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  IconChevronLeft,
  IconPlus,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import { ClientOnly } from '@/components/client-only'
import { NotificationBox } from '@/components/notification-box'
import { ThemeToggle } from '@/components/theme-toggle'
import { MyAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import { SearchContext } from './search-context'

const deleteThreadMutation = graphql(/* GraphQL */ `
  mutation DeleteThread($id: ID!) {
    deleteThread(id: $id)
  }
`)

type HeaderProps = {
  threadIdFromURL?: string
  streamingDone?: boolean
  threadId?: string
}

export function Header({
  threadIdFromURL,
  streamingDone,
  threadId
}: HeaderProps) {
  const router = useRouter()
  const { isThreadOwner } = useContext(SearchContext)
  const [deleteAlertVisible, setDeleteAlertVisible] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const [enablePage] = useEnablePage()

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
    e.preventDefault()
    setIsDeleting(true)
    deleteThread({
      id: threadIdFromURL!
    })
  }

  const onNavigateToHomePage = (scroll?: boolean) => {
    if (scroll) {
      clearHomeScrollPosition()
    }
    router.push('/')
  }

  const onConvertToPage = () => {
    if (!threadId) return
    updatePendingThreadId(threadId)
    router.push('/pages')
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
            className="flex items-center gap-1 px-2 font-normal text-muted-foreground"
            onClick={() => onNavigateToHomePage(true)}
          >
            <IconPlus />
          </Button>
        )}

        {streamingDone && threadIdFromURL && isThreadOwner && (
          <AlertDialog
            open={deleteAlertVisible}
            onOpenChange={setDeleteAlertVisible}
          >
            <AlertDialogTrigger asChild>
              <Button size="icon" variant="hover-destructive">
                <IconTrash />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete this thread</AlertDialogTitle>
                <AlertDialogDescription>
                  Are you sure you want to delete this thread? This operation is
                  not revertible.
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
        )}
        {!!enablePage.value && streamingDone && isThreadOwner && threadId && (
          <Button variant="ghost" onClick={onConvertToPage} className="gap-1">
            Convert to page
          </Button>
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
    </header>
  )
}
