const OAUTH_PROVIDERS = ['github', 'google'] as const

type OauthProvider = typeof OAUTH_PROVIDERS[number]


export { OAUTH_PROVIDERS }
export type { OauthProvider }
