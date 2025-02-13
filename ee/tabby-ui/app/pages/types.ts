import {
  ListPageSectionsQuery,
  ListPagesQuery
} from '@/lib/gql/generates/graphql'

export type PageItem = ListPagesQuery['pages']['edges'][0]['node']
export type SectionItem = Omit<
  ListPageSectionsQuery['pageSections']['edges'][0]['node'],
  '__typename'
>
