'use client'

import { useRouter } from 'next/navigation'

import CrawlerForm from './create-crawler-form'

export const NewCrawler = () => {
  const router = useRouter()

  const onCreated = () => {
    router.back()
  }

  return (
    <>
      <CrawlerForm onCreated={onCreated} />
    </>
  )
}
