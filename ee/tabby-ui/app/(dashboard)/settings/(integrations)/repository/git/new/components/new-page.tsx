'use client'

import { useRouter } from 'next/navigation'

import RepositoryForm from './create-repository-form'

export const NewRepository = () => {
  const router = useRouter()

  const onCreated = () => {
    router.back()
  }

  return (
    <>
      <RepositoryForm onCreated={onCreated} />
    </>
  )
}
