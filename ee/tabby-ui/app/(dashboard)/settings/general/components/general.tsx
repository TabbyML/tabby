'use client'

import React from 'react'

import { Separator } from '@/components/ui/separator'
import { ListSkeleton } from '@/components/skeleton'

import { GeneralFormSection } from './form-section'
import { GeneralNetworkForm } from './network-form'
import { GeneralSecurityForm } from './security-form'

export default function General() {
  const [initialized, setInitialized] = React.useState(false)

  React.useEffect(() => {
    setTimeout(() => {
      setInitialized(true)
    }, 500)
  }, [])

  if (!initialized) return <ListSkeleton />

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
