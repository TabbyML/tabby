'use client'

import { useSearchParams } from 'next/navigation'

import { PasswordResetForm } from './password-reset-form'

export default function ResetPassword() {
  const searchParams = useSearchParams()
  const code = searchParams.get('code') || undefined

  return (
    <div className="w-[350px] space-y-6">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">
          Reset Password
        </h1>
        <p className="text-sm text-muted-foreground">Enter a new password</p>
      </div>
      <PasswordResetForm code={code} />
    </div>
  )
}
