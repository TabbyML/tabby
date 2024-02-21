'use client'

import React from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'

import { buttonVariants } from '@/components/ui/button'
import { IconCheckCircled } from '@/components/ui/icons'

import { ResetPasswordForm } from './reset-password-form'

export default function ResetPasswordPage() {
  const [resetSuccess, setResetSuccess] = React.useState(false)
  const searchParams = useSearchParams()
  const code = searchParams.get('code') || undefined

  if (resetSuccess) {
    return (
      <div className="w-[350px] space-y-6">
        <div className="flex flex-col space-y-2 text-center">
          <div className="flex justify-center">
            <IconCheckCircled className="h-12 w-12 text-successful-foreground" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Set a new password
          </h1>
          <p className="pb-4 text-sm text-muted-foreground">
            Your password has been changed.
          </p>
          <Link replace href="/auth/signin" className={buttonVariants()}>
            Go to Sign In
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="w-[350px] space-y-6">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">
          Set a new password
        </h1>
        <p className="text-sm text-muted-foreground">
          Please enter a new password for your account.
        </p>
      </div>
      <ResetPasswordForm code={code} onSuccess={() => setResetSuccess(true)} />
    </div>
  )
}
