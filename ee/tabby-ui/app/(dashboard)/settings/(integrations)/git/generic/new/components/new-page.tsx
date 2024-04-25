'use client'

import { useRouter } from 'next/navigation'

import { Button } from '@/components/ui/button'
import { CardTitle } from '@/components/ui/card'
import { IconChevronLeft } from '@/components/ui/icons'

import RepositoryForm from './create-repository-form'

export const NewRepository = () => {
  const router = useRouter()

  const onCreated = () => {
    router.replace('/settings/git/generic')
  }

  return (
    <>
      <CardTitle className="py-6">
        <div className="-ml-1 flex items-center">
          <Button
            onClick={() => router.back()}
            variant={'ghost'}
            className="h-6 px-1"
          >
            <IconChevronLeft className="h-5 w-5" />
          </Button>
          <span className="ml-2">Create Generic git repository</span>
        </div>
      </CardTitle>
      <RepositoryForm onCreated={onCreated} />
    </>
  )
}
