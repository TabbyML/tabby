'use client'

import Link from 'next/link'

import { buttonVariants } from '@/components/ui/button'

import RepositoryTable from './repository-table'

export default function Git() {
  return (
    <>
      <div className="my-4 flex justify-end">
        <Link href="./git/new" className={buttonVariants()}>
          Create
        </Link>
      </div>
      <RepositoryTable />
    </>
  )
}
