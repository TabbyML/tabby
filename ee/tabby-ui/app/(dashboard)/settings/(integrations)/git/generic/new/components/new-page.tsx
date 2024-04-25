'use client'

import { useRouter } from 'next/navigation'

import RepositoryForm from './create-repository-form'

export const NewRepository = () => {
  const router = useRouter()

  const onCreated = () => {
    router.replace('/settings/git/generic')
  }

  return <RepositoryForm onCreated={onCreated} />
}
