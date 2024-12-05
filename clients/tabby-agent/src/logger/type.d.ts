export interface Logger {
  error: (msg: string, error: any) => void;
  warn: (msg: string) => void;
  info: (msg: string) => void;
  debug: (msg: string) => void;
  trace: (msg: string, verbose?: any) => void;
}
