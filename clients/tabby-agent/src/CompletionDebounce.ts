import { CancelablePromise } from "./generated";
import { CompletionRequest } from "./Agent";
import { AgentConfig } from "./AgentConfig";
import { rootLogger } from "./logger";
import { splitLines } from "./utils";

function clamp(min: number, max: number, value: number): number {
  return Math.max(min, Math.min(max, value));
}

export class CompletionDebounce {
  private readonly logger = rootLogger.child({ component: "CompletionDebounce" });
  private ongoing: CancelablePromise<any> | null = null;
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

  debounce<T>(
    callback: () => T | CancelablePromise<T>,
    context: {
      request: CompletionRequest;
      config: AgentConfig["completion"]["debounce"];
      responseTime: number;
    },
  ): CancelablePromise<T> {
    const { request, config, responseTime } = context;
    if (request.manually) {
      return this.renewPromise(0, callback);
    }
    if (config.mode === "fixed") {
      return this.renewPromise(config.interval, callback);
    }
    const now = Date.now();
    this.updateBaseInterval(now - this.lastCalledTimeStamp);
    this.lastCalledTimeStamp = now;
    const contextScore = this.calcContextScore(request);
    const adaptiveRate =
      this.options.adaptiveRate.max - (this.options.adaptiveRate.max - this.options.adaptiveRate.min) * contextScore;
    const expectedLatency = adaptiveRate * this.baseInterval;
    const delay = clamp(this.options.requestDelay.min, this.options.requestDelay.max, expectedLatency - responseTime);
    return this.renewPromise(delay, callback);
  }

  private renewPromise<T>(delay: number, callback: () => T | CancelablePromise<T>): CancelablePromise<T> {
    if (this.ongoing) {
      this.ongoing.cancel();
    }
    this.ongoing = new CancelablePromise<T>((resolve, reject, onCancel) => {
      let cancelable: CancelablePromise<T> | null = null;
      const timer = setTimeout(
        async () => {
          const result = callback();
          if (result instanceof CancelablePromise) {
            cancelable = result;
          }
          resolve(await result);
        },
        Math.min(delay, 0x7fffffff),
      );
      onCancel(() => {
        cancelable?.cancel();
        clearTimeout(timer);
      });
    });
    return this.ongoing;
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
