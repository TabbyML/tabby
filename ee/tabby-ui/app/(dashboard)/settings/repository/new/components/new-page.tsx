'use client'

import { useRouter } from 'next/navigation'

import RepositoryForm from '../../components/create-repository-form'
import { RepositoryHeader } from '../../components/header'

export const NewRepository = () => {
  const router = useRouter()

  const onCreated = () => {
    router.replace('/settings/repository')
  }

  return (
    <>
      <RepositoryHeader />
      <RepositoryForm onCreated={onCreated} />
    </>
  )
}
