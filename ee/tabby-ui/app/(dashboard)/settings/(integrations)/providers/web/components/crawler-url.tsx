'use client'

import Link from 'next/link'

import { buttonVariants } from '@/components/ui/button'

import CrawlerUrlTable from './crawler-url-table'

export default function CrawlerUrl() {
  return (
    <>
      <div className="mt-4 flex justify-end">
        <Link href="./web/new" className={buttonVariants()}>
          Create
        </Link>
      </div>
      <CrawlerUrlTable />
    </>
  )
}
