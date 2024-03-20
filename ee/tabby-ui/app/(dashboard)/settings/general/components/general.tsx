'use client'

import React from 'react'

import { Separator } from '@/components/ui/separator'

import { GeneralFormSection } from './form-section'
import { GeneralNetworkForm } from './network-form'
import { GeneralSecurityForm } from './security-form'

export default function General() {
  return (
    <div className="flex flex-col gap-4">
      <GeneralFormSection title="Network">
        <GeneralNetworkForm />
      </GeneralFormSection>
      <Separator className="mb-8" />
      <GeneralFormSection title="Security">
        <GeneralSecurityForm />
      </GeneralFormSection>
    </div>
  )
}
