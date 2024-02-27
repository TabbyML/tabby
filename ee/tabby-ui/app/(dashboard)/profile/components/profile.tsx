'use client'

import React from 'react'

import { ProfileCard } from './profile-card'

export default function Profile() {
  return (
    <div className="flex flex-col gap-4">
      <ProfileCard title="Password">
        <div>password form</div>
      </ProfileCard>
    </div>
  )
}
