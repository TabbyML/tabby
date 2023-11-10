export type ISearchHit = {
  id: number
  doc?: {
    body?: string
    name?: string
    filepath?: string
    git_url?: string
    kind?: string
    language?: string
  }
}
export type SearchReponse = {
  hits?: Array<ISearchHit>
  num_hits?: number
}
