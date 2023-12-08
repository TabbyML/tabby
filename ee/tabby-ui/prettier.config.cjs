/** @type {import('prettier').Config} */
module.exports = {
  endOfLine: 'lf',
  semi: false,
  useTabs: false,
  singleQuote: true,
  arrowParens: 'avoid',
  tabWidth: 2,
  trailingComma: 'none',
  importOrder: [
    '^(react/(.*)$)|^(react$)',
    '^(next/(.*)$)|^(next$)',
    '<THIRD_PARTY_MODULES>',
    '',
    '^types$',
    '^@/types/(.*)$',
    '^@/config/(.*)$',
    '^@/lib/(.*)$',
    '^@/hooks/(.*)$',
    '^@/components/ui/(.*)$',
    '^@/components/(.*)$',
    '^@/registry/(.*)$',
    '^@/styles/(.*)$',
    '^@/app/(.*)$',
    '',
    '^[./]'
  ],
  importOrderParserPlugins: ['typescript', 'jsx', 'decorators-legacy'],
}
