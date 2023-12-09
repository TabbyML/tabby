'use client'

import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { Separator } from '@/components/ui/separator'
import SlackDialog from '@/components/slack-dialog'

export default function Home() {
  return (
    <div className="p-4 lg:p-16">
      <MainPanel />
      <SlackDialog />
    </div>
  )
}

function toBadgeString(str: string) {
  return encodeURIComponent(str.replaceAll('-', '--'))
}

function MainPanel() {
  const { data: healthInfo } = useHealth()

  if (!healthInfo) return

  return (
    <div className="flex w-full flex-col gap-3">
      <h1>
        <span className="font-bold">Congratulations</span>, your tabby instance
        is up!
      </h1>
      <span className="flex flex-wrap gap-1">
        <a
          target="_blank"
          href={`https://github.com/TabbyML/tabby/releases/tag/${healthInfo.version.git_describe}`}
        >
          <img
            src={`https://img.shields.io/badge/version-${toBadgeString(
              healthInfo.version.git_describe
            )}-green`}
          />
        </a>
      </span>
      <Separator />
    </div>
  )
}
