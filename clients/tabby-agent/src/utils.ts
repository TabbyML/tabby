export function sleep(milliseconds: number) {
  return new Promise((r) => setTimeout(r, milliseconds));
}

export function splitLines(input: string) {
  return input.match(/.*(?:$|\r?\n)/g).filter(Boolean) // Split lines and keep newline character
}
