'use client'

import { useRouter } from 'next/navigation'
import { toast } from 'sonner'

import RepositoryForm from '../../components/create-repository-form'

export const NewRepository = () => {
  const router = useRouter()

  const onCreated = () => {
    toast.success('Git Repo created')
    router.replace('/settings/repository')
  }

  return (
    <div className="p-4">
      <div className="mb-4 flex items-center gap-4 min-h-8">
        <div className="flex-1 text-sm text-muted-foreground">
          Git Repository
        </div>
      </div>
      <RepositoryForm onCreated={onCreated} />
    </div>
  )
}
