export type TFile = {
  kind: 'file' | 'dir'
  basename: string
}

export type TRange = { start: number; end: number }
export type TPointRange = { start: TPoint; end: TPoint }
export type TPoint = { row: number; column: number }

export type TCodeTag = {
  range: TRange
  name_range: TRange
  line_range: TRange
  is_definition: boolean
  syntax_type_name: string
  utf16_column_range: TRange
  span: TPointRange
}

export type ResolveEntriesResponse = { entries: TFile[] }
