'use client'

import Link from 'next/link'

import { buttonVariants } from '@/components/ui/button'

import { RepositoryHeader } from './header'
import RepositoryTable from './repository-table'

export default function Repository() {
  return (
    <>
      <RepositoryHeader
        extra={
          <Link href="/settings/repository/new" className={buttonVariants()}>
            Add Git Repo
          </Link>
        }
      />
      <RepositoryTable />
    </>
  )
}
