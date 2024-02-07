'use client'

import React from 'react'

import { ListSkeleton } from '@/components/skeleton'

import { GeneralFormSection } from './form-section'
import { GeneralNetworkForm } from './network-form'
import { GeneralSecurityForm } from './security-form'
import { Separator } from '@/components/ui/separator'

export default function General() {
  // todo usequery

  const [initialized, setInitialized] = React.useState(false)

  React.useEffect(() => {
    setTimeout(() => {
      // get data from query and then setInitialized
      setInitialized(true)
    }, 500)
  }, [])

  // makes it convenient to set the defaultValues of forms
  if (!initialized) return <ListSkeleton />

  return (
    <div className="flex flex-col gap-4">
      <GeneralFormSection title="Network">
        {/* todo pass defualtValues from useQuery */}
        <GeneralNetworkForm />
      </GeneralFormSection>
      <Separator className='mb-8' />
      <GeneralFormSection title="Security">
        {/* todo pass defualtValues from useQuery */}
        <GeneralSecurityForm />
      </GeneralFormSection>
    </div>
  )
}
