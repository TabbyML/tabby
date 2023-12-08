'use client'

import UserSignInForm from './user-signin-form'

export default function Signin() {
  return (
    <div className="w-[350px] space-y-6">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">Sign In</h1>
        <p className="text-sm text-muted-foreground">
          Enter credentials to login to your account
        </p>
      </div>
      <UserSignInForm />
    </div>
  )
}
