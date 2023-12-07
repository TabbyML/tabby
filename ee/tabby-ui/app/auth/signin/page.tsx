'use client'

import UserSignInForm from "./components/user-signin-form"

export default function Signin() {
  return (
    <div className="space-y-6 w-[350px]">
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
