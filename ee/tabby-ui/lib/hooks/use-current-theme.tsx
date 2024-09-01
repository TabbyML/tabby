import { useTheme } from 'next-themes'

export function useCurrentTheme() {
  const { theme, systemTheme, setTheme } = useTheme()

  let currentTheme
  if (theme && theme !== 'system') {
    currentTheme = theme
  } else {
    currentTheme = systemTheme || 'light'
  }

  return {
    theme: currentTheme,
    setTheme,
    systemTheme
  }
}
