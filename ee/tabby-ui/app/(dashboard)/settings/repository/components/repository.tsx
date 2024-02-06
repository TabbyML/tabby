'use client'

import Link from 'next/link'

import { buttonVariants } from '@/components/ui/button'

import { RepositoryHeader } from './header'
import RepositoryTable from './repository-table'

export default function Repository() {
  return (
    <>
      <RepositoryHeader />
      <RepositoryTable />
      <div className="mt-4 flex justify-end">
        <Link href="/settings/repository/new" className={buttonVariants()}>
          Create
        </Link>
      </div>
    </>
  )
}
