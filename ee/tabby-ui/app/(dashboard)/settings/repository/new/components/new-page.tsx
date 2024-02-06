'use client'

import { useRouter } from 'next/navigation'
import { toast } from 'sonner'

import RepositoryForm from '../../components/create-repository-form'
import { RepositoryHeader } from '../../components/header'

export const NewRepository = () => {
  const router = useRouter()

  const onCreated = () => {
    toast.success('Git Repo created')
    router.replace('/settings/repository')
  }

  return (
    <>
      <RepositoryHeader />
      <RepositoryForm onCreated={onCreated} />
    </>
  )
}
