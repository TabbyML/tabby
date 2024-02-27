'use client'

import React from 'react'

import { ChangePassword } from './change-password'
import { ProfileCard } from './profile-card'

export default function Profile() {
  return (
    <div className="flex flex-col gap-6">
      <ProfileCard title="Password">
        <ChangePassword />
      </ProfileCard>
    </div>
  )
}
