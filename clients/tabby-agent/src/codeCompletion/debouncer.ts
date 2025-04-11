import type { ConfigData } from "../config/type";

function clamp(min: number, max: number, value: number): number {
  return Math.max(min, Math.min(max, value));
}

export type DebouncingContext = {
  triggerCharacter: string;
  isLineEnd?: boolean;
  isDocumentEnd?: boolean;
  manually?: boolean;
  estimatedResponseTime?: number;
};

export class CompletionDebouncer {
  private baseInterval = 200; // ms
  private lastTimestamp = 0;
  private intervalHistory: number[] = [];
  private config: ConfigData["completion"]["debounce"] | undefined = undefined;
  private rules = {
    baseIntervalSlideWindowAvg: {
      minSize: 20,
      maxSize: 100,
      min: 100,
      max: 400,
    },
    adaptiveRate: {
      min: 1.5,
      max: 3.0,
    },
    contextScoreWeights: {
      triggerCharacter: 0.5,
      lineEnd: 0.4,
      documentEnd: 0.1,
    },
    requestDelay: {
      min: 100, // ms
      max: 1000,
    },
  };

  updateConfig(config: ConfigData["completion"]["debounce"] | undefined) {
    this.config = config;
  }

  async debounce(context: DebouncingContext, signal?: AbortSignal): Promise<void> {
    if (context.manually) {
      return this.sleep(0, signal);
    }
    if (this.config?.mode === "fixed") {
      return this.sleep(this.config.interval, signal);
    }
    const now = Date.now();
    this.updateBaseInterval(now - this.lastTimestamp);
    this.lastTimestamp = now;
    const contextScore = this.calcContextScore(context);
    const adaptiveRate =
      this.rules.adaptiveRate.max - (this.rules.adaptiveRate.max - this.rules.adaptiveRate.min) * contextScore;
    const expectedLatency = adaptiveRate * this.baseInterval;
    const responseTime = context.estimatedResponseTime ?? 0;
    const delay = clamp(this.rules.requestDelay.min, this.rules.requestDelay.max, expectedLatency - responseTime);
    return this.sleep(delay, signal);
  }

  private async sleep(delay: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, Math.min(delay, 0x7fffffff));
      if (signal) {
        if (signal.aborted) {
          clearTimeout(timer);
          reject(signal.reason);
        } else {
          signal.addEventListener("abort", () => {
            clearTimeout(timer);
            reject(signal.reason);
          });
        }
      }
    });
  }

  private updateBaseInterval(interval: number) {
    if (interval > this.rules.baseIntervalSlideWindowAvg.max) {
      return;
    }
    this.intervalHistory.push(interval);
    if (this.intervalHistory.length > this.rules.baseIntervalSlideWindowAvg.maxSize) {
      this.intervalHistory.shift();
    }
    if (this.intervalHistory.length > this.rules.baseIntervalSlideWindowAvg.minSize) {
      const avg = this.intervalHistory.reduce((a, b) => a + b, 0) / this.intervalHistory.length;
      this.baseInterval = clamp(
        this.rules.baseIntervalSlideWindowAvg.min,
        this.rules.baseIntervalSlideWindowAvg.max,
        avg,
      );
    }
  }

  // return score in [0, 1], 1 means the context has a high chance to accept the completion
  private calcContextScore(context: DebouncingContext): number {
    const { triggerCharacter, isLineEnd, isDocumentEnd } = context;
    const weights = this.rules.contextScoreWeights;
    let score = 0;
    score += triggerCharacter.match(/^\W*$/) ? weights.triggerCharacter : 0;
    score += isLineEnd ? weights.lineEnd : 0;
    score += isDocumentEnd ? weights.documentEnd : 0;
    score = clamp(0, 1, score);
    return score;
  }
}
