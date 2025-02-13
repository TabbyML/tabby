'use client'

import { useContext, useState } from 'react'
import type { MouseEvent } from 'react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
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
  IconChevronLeft,
  IconEdit,
  IconMore,
  IconShare,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import { ClientOnly } from '@/components/client-only'
import { NotificationBox } from '@/components/notification-box'
import { ThemeToggle } from '@/components/theme-toggle'
import { MyAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import { PageContext } from './page-context'

const deletePageMutation = graphql(/* GraphQL */ `
  mutation DeletePage($id: ID!) {
    deletePage(id: $id)
  }
`)

type HeaderProps = {
  pageIdFromURL?: string
  streamingDone?: boolean
}

export function Header({ pageIdFromURL, streamingDone }: HeaderProps) {
  const router = useRouter()
  const { isPageOwner, mode, setMode, isLoading } = useContext(PageContext)
  const isEditMode = mode === 'edit'
  const [deleteAlertVisible, setDeleteAlertVisible] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const { copyToClipboard } = useCopyToClipboard({})

  const deletePage = useMutation(deletePageMutation, {
    onCompleted(data) {
      if (data.deletePage) {
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

  const handleDeletePage = (e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault()
    setIsDeleting(true)
    deletePage({
      id: pageIdFromURL!
    }).then(data => {
      if (data?.data?.deletePage) {
        router.replace('/')
      } else {
        toast.error('Failed to delete')
        setIsDeleting(false)
      }
    })
  }

  const onNavigateToHomePage = (scroll?: boolean) => {
    if (scroll) {
      clearHomeScrollPosition()
    }
    router.push('/')
  }

  const onShare = () => {
    if (typeof window === 'undefined') return
    copyToClipboard(window.location.href)
  }

  return (
    <header className="relative flex h-16 w-full items-center justify-between border-b px-4 lg:px-10">
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
        {isPageOwner ? (
          <>
            {!isEditMode ? (
              <>
                <DropdownMenu modal={false}>
                  <DropdownMenuTrigger asChild>
                    <Button size="icon" variant="ghost">
                      <IconMore />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      className="cursor-pointer gap-2"
                      onSelect={onShare}
                    >
                      <IconShare />
                      Share
                    </DropdownMenuItem>
                    {streamingDone && pageIdFromURL && isPageOwner && (
                      <DropdownMenuItem
                        className="cursor-pointer gap-2 !text-destructive"
                        onSelect={() => setDeleteAlertVisible(true)}
                      >
                        <IconTrash />
                        Delete Page
                      </DropdownMenuItem>
                    )}
                  </DropdownMenuContent>
                </DropdownMenu>

                <Button
                  variant="ghost"
                  className="flex items-center gap-1 px-2 font-normal"
                  onClick={() => setMode('edit')}
                >
                  <IconEdit />
                  Edit
                </Button>
              </>
            ) : (
              <>
                <Button
                  disabled={!streamingDone}
                  onClick={e => setMode('view')}
                >
                  Done
                </Button>
              </>
            )}
          </>
        ) : null}
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

      {/* status badge */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
        {isLoading ? (
          <Badge variant="secondary">Writing...</Badge>
        ) : mode === 'edit' ? (
          <Badge>Editing</Badge>
        ) : null}
      </div>

      <AlertDialog
        open={deleteAlertVisible}
        onOpenChange={setDeleteAlertVisible}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this page</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this page? This operation is not
              revertible.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className={buttonVariants({
                variant: 'destructive'
              })}
              onClick={handleDeletePage}
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
