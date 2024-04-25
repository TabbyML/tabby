'use client'

import Link from 'next/link'

import { buttonVariants } from '@/components/ui/button'
import { CardTitle } from '@/components/ui/card'

import RepositoryTable from './repository-table'

export default function Repository() {
  return (
    <>
      <CardTitle className="py-3">Generic git repositories</CardTitle>
      <RepositoryTable />
      <div className="mt-4 flex justify-end">
        <Link href="/settings/git/generic/new" className={buttonVariants()}>
          Create
        </Link>
      </div>
    </>
  )
}
