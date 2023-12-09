'use client'

import { useEffect, useState } from 'react'

import { useSession } from '@/lib/tabby/auth'
import { buttonVariants } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { IconSlack } from '@/components/ui/icons'

const COMMUNITY_DIALOG_SHOWN_KEY = 'community-dialog-shown'

export default function SlackDialog() {
  const { status } = useSession()
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (status !== 'authenticated') return

    if (!localStorage.getItem(COMMUNITY_DIALOG_SHOWN_KEY)) {
      setOpen(true)
      localStorage.setItem(COMMUNITY_DIALOG_SHOWN_KEY, 'true')
    }
  }, [status])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogHeader className="gap-3">
          <DialogTitle>Join the Tabby community</DialogTitle>
          <DialogDescription>
            Connect with other contributors building Tabby. Share knowledge, get
            help, and contribute to the open-source project.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="sm:justify-start">
          <a
            target="_blank"
            href="https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA"
            className={buttonVariants()}
          >
            <IconSlack className="-ml-2 h-8 w-8" />
            Join us on Slack
          </a>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
