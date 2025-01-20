'use client'

import React from 'react'

import { Avatar } from './avatar'
import { ChangeName } from './change-name'
import { ChangePassword } from './change-password'
import { Email } from './email'
import { ProfileCard } from './profile-card'

export default function Profile() {
  return (
    <div className="flex flex-col items-center gap-6">
      <ProfileCard title="Your Name" footerClassname="pb-0">
        <ChangeName />
      </ProfileCard>
      <ProfileCard
        title="Your Email"
        description="This will be the email you use to log in and receive notifications."
        footer="The feature to change your email address will be available in a future release."
      >
        <Email />
      </ProfileCard>
      <ProfileCard
        title="Your Avatar"
        description="This is your avatar image."
        footerClassname="pb-0"
      >
        <Avatar />
      </ProfileCard>
      <ProfileCard
        title="Change Password"
        footerClassname="pb-0"
        hideForSsoUser
      >
        <ChangePassword />
      </ProfileCard>
    </div>
  )
}
