'use client'

import React from 'react'

import { Button, ButtonProps } from '@/components/ui/button'
import { IconCheck, IconTrash } from '@/components/ui/icons'

interface ClearChatsButtonProps extends ButtonProps {
  onClear: () => void
}

export const ClearChatsButton = ({
  onClear,
  onClick,
  onBlur,
  ...rest
}: ClearChatsButtonProps) => {
  const [waitingConfirmation, setWaitingConfirmation] = React.useState(false)

  const cancelConfirmation = () => {
    setWaitingConfirmation(false)
  }

  const handleBlur: React.FocusEventHandler<HTMLButtonElement> = e => {
    if (waitingConfirmation) {
      cancelConfirmation()
    }
    onBlur?.(e)
  }

  const handleClick: React.MouseEventHandler<HTMLButtonElement> = e => {
    if (!waitingConfirmation) {
      setWaitingConfirmation(true)
    } else {
      onClear()
      setWaitingConfirmation(false)
    }
    onClick?.(e)
  }

  return (
    <Button
      className="h-12 w-full justify-start"
      variant="ghost"
      {...rest}
      onClick={handleClick}
      onBlur={handleBlur}
    >
      {waitingConfirmation ? (
        <>
          <IconCheck />
          <span className="ml-2">Confirm Clear Chats</span>
        </>
      ) : (
        <>
          <IconTrash />
          <span className="ml-2">Clear Chats</span>
        </>
      )}
    </Button>
  )
}
