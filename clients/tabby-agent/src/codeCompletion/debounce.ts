import type { CompletionRequest } from "./contexts";
import type { ConfigData } from "../config/type";
import { splitLines } from "../utils/string";

function clamp(min: number, max: number, value: number): number {
  return Math.max(min, Math.min(max, value));
}

export class CompletionDebounce {
  private lastCalledTimeStamp = 0;

  private baseInterval = 200; // ms
  private calledIntervalHistory: number[] = [];

  private options = {
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
      noSuffixInCurrentLine: 0.4,
      noSuffix: 0.1,
    },
    requestDelay: {
      min: 100, // ms
      max: 1000,
    },
  };

  async debounce(
    context: {
      request: CompletionRequest;
      config: ConfigData["completion"]["debounce"];
      responseTime: number;
    },
    signal?: AbortSignal,
  ): Promise<void> {
    const { request, config, responseTime } = context;
    if (request.manually) {
      return this.sleep(0, signal);
    }
    if (config.mode === "fixed") {
      return this.sleep(config.interval, signal);
    }
    const now = Date.now();
    this.updateBaseInterval(now - this.lastCalledTimeStamp);
    this.lastCalledTimeStamp = now;
    const contextScore = this.calcContextScore(request);
    const adaptiveRate =
      this.options.adaptiveRate.max - (this.options.adaptiveRate.max - this.options.adaptiveRate.min) * contextScore;
    const expectedLatency = adaptiveRate * this.baseInterval;
    const delay = clamp(this.options.requestDelay.min, this.options.requestDelay.max, expectedLatency - responseTime);
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
    if (interval > this.options.baseIntervalSlideWindowAvg.max) {
      return;
    }
    this.calledIntervalHistory.push(interval);
    if (this.calledIntervalHistory.length > this.options.baseIntervalSlideWindowAvg.maxSize) {
      this.calledIntervalHistory.shift();
    }
    if (this.calledIntervalHistory.length > this.options.baseIntervalSlideWindowAvg.minSize) {
      const avg = this.calledIntervalHistory.reduce((a, b) => a + b, 0) / this.calledIntervalHistory.length;
      this.baseInterval = clamp(
        this.options.baseIntervalSlideWindowAvg.min,
        this.options.baseIntervalSlideWindowAvg.max,
        avg,
      );
    }
  }

  // return score in [0, 1], 1 means the context has a high chance to accept the completion
  private calcContextScore(request: CompletionRequest): number {
    let score = 0;
    const weights = this.options.contextScoreWeights;
    const triggerCharacter = request.text[request.position - 1] ?? "";
    score += triggerCharacter.match(/^\W*$/) ? weights.triggerCharacter : 0;

    const suffix = request.text.slice(request.position) ?? "";
    const currentLineInSuffix = splitLines(suffix)[0] ?? "";
    score += currentLineInSuffix.match(/^\W*$/) ? weights.noSuffixInCurrentLine : 0;
    score += suffix.match(/^\W*$/) ? weights.noSuffix : 0;

    score = clamp(0, 1, score);
    return score;
  }
}
