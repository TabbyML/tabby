'use client'

import React from 'react'

import { Avatar } from './avatar'
import { ChangePassword } from './change-password'
import { Email } from './email'
import { ProfileCard } from './profile-card'

export default function Profile() {
  return (
    <div className="flex flex-col gap-6">
      <ProfileCard
        title="Your Avatar"
        description="This is your avatar image on Tabby"
        footer="Square image recommended."
      >
        <Avatar />
      </ProfileCard>
      <ProfileCard
        title="Your Email"
        description="This is the email associated with your Tabby account"
        footer="Your email"
      >
        <Email />
      </ProfileCard>
      <ProfileCard title="Change Password" footerClassname="pb-0">
        <ChangePassword />
      </ProfileCard>
    </div>
  )
}
