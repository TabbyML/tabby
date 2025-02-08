import {
  ListPageSectionsQuery,
  ListPagesQuery
} from '@/lib/gql/generates/graphql'

export type PageItem = ListPagesQuery['pages']['edges'][0]['node']
export type SectionItem =
  ListPageSectionsQuery['pageSections']['edges'][0]['node']
